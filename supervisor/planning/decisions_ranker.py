"""Bounded Decisions API candidate ranking; never blocks the controller tick.

Previews describe proposed tool motion, not predicted object outcomes. This module
never reads simulator object truth, trains a model, or asserts task completion.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from .m7_service import Guard, log_value

ENDPOINT = 'https://api.openai.com/v1/decisions'
MODEL = 'gpt-6-luna'


def project(points, camera):
    if 'quat_wxyz' in camera:
        q = np.asarray(camera['quat_wxyz'])
        matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    else:
        back = np.asarray(camera['position_m']) - camera['look_at_m']
        back /= np.linalg.norm(back)
        right = np.cross([0., 0., 1.], back); right /= np.linalg.norm(right)
        matrix = np.column_stack([right, np.cross(back, right), back])
    local = (np.asarray(points) - camera['position_m']) @ matrix
    if np.any(local[:, 2] >= -.001):
        raise ValueError('preview_point_behind_camera')
    k = np.asarray(camera['intrinsics_px'])
    return np.column_stack([k[0, 2] + k[0, 0]*local[:, 0]/-local[:, 2],
                            k[1, 2] - k[1, 1]*local[:, 1]/-local[:, 2]])


def candidate_card(observation, actions, label, size=320):
    """Calibrated tool-path glyphs on current RGB; no simulated future frames."""
    parts = []
    for name in ['primary', 'secondary']:
        im = Image.fromarray(observation['rgb'][name]).convert('RGB')
        d = ImageDraw.Draw(im)
        path = np.vstack([observation['proprio']['eef_position_m'], actions[:, :3]])
        xy = project(path, observation['cameras'][name])
        d.line([tuple(x) for x in xy], fill=(10, 150, 255), width=4)
        for i in sorted(set([0, len(actions)//2, len(actions)-1])):
            a = actions[i]; x = a[3:6]/np.linalg.norm(a[3:6])
            y = a[6:9] - np.dot(a[6:9], x)*x; y /= np.linalg.norm(y)
            # Tool orientation glyph, not an IK solution or collision forecast.
            points = np.array([a[:3]-.022*x, a[:3]+.022*x, a[:3], a[:3]+.03*y])
            pix = project(points, observation['cameras'][name])
            d.line([tuple(pix[0]), tuple(pix[1])], fill=(30, 180, 255), width=4)
            d.line([tuple(pix[2]), tuple(pix[3])], fill=(100, 220, 255), width=4)
            d.text(tuple(pix[2]), str(i+1), fill='white', stroke_width=1, stroke_fill='black')
        im.thumbnail((size, size))
        parts.append(im)
    card = Image.new('RGB', (size*2, size+34), '#111a24')
    card.paste(parts[0], (0, 34)); card.paste(parts[1], (size, 34))
    ImageDraw.Draw(card).text((8, 8), label+' | tool targets only; object outcome unknown', fill='white')
    return card


def _image(im):
    buf = io.BytesIO(); im.save(buf, format='JPEG', quality=88)
    return {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()}


def payload(observation, candidates):
    labels = list(candidates)
    if not 2 <= len(labels) <= 8:
        raise ValueError('rank_two_to_eight_candidates')
    current = Image.fromarray(np.hstack([observation['rgb']['primary'], observation['rgb']['secondary']]))
    current.thumbnail((1024, 512))
    cards = [candidate_card(observation, a, label) for label, a in candidates.items()]
    panel = Image.new('RGB', (1280, math.ceil(len(cards)/2)*354), '#111a24')
    for i, card in enumerate(cards): panel.paste(card, ((i%2)*640, (i//2)*354))
    text = ('Select one alternative proposed Panda tool trajectory for the CURRENT scene. '
            'First image: current overhead and oblique views. Second: independent candidate cards. '
            'Blue paths and orientation glyphs are requested future tool poses, not actual future robot '
            'or object positions. Numbers are control steps at 20Hz. Current objects are unchanged in '
            'every card. Finger closure is a command, never proof of a grasp. Compare approach, alignment, '
            'retaining an apparent grasp, and progress toward the supplied task. Reject obviously unhelpful '
            'alternatives; choose UNKNOWN if the images cannot distinguish useful progress. '
            'Do not assume an unseen successful grasp, object rotation, or completed placement. '
            'Task: '+str(observation['task']['instruction'])+'\n'+
            'Gripper close probabilities (first, middle, last): '+json.dumps({k:[float(a[i,9]) for i in [0,len(a)//2,len(a)-1]] for k,a in candidates.items()}))
    body = {'model': MODEL, 'input': [{'role': 'user', 'content': [
        {'type':'input_text','text':text}, _image(current), _image(panel)]}],
        'questions':[{'type':'choice','name':'best_candidate',
            'instructions':'Choose the most useful admissible next motion. UNKNOWN is allowed. Candidate labels imply no rank.',
            'choices':[{'value':k,'description':'Candidate '+k} for k in labels]+[{'value':'UNKNOWN','description':'No reliably preferable useful motion is visible.'}]}]}
    return body, panel


def parse(response, labels):
    if response.get('model') != MODEL: raise ValueError('wrong_model_identity')
    answers = response.get('answers', [])
    if len(answers) != 1 or answers[0].get('name') != 'best_candidate': raise ValueError('wrong_answer_identity')
    a = answers[0]
    if a.get('type') == 'refusal': return {'choice':'UNKNOWN','status':'refused'}
    allowed = set(labels)|{'UNKNOWN'}
    if a.get('type') != 'choice' or a.get('choice') not in allowed: raise ValueError('invalid_choice')
    probs = a.get('probabilities', [])
    if len(probs) != len(allowed) or {p['value'] for p in probs} != allowed: raise ValueError('invalid_probability_labels')
    if any(not math.isfinite(p['probability']) or not 0 <= p['probability'] <= 1 for p in probs): raise ValueError('invalid_probability')
    if abs(sum(p['probability'] for p in probs)-1) >= .05: raise ValueError('invalid_probability_total')
    return {'choice':a['choice'],'status':'valid','probabilities':probs,'confidence':a.get('confidence')}


class DecisionsRanker:
    """One in-flight comparison of up to eight alternatives; poll() never waits.

    The caller keeps executing a previously admitted chunk, rechecks guards every
    tick, and supplies a context token identifying its episode and plan version.
    A result can only select a proposal; it cannot bypass controller validation.
    """
    def __init__(self, config, log_dir, deadline_s=.5, transport_timeout_s=2., transport=None, clock=time.monotonic):
        if not 0 < deadline_s <= transport_timeout_s <= 10: raise ValueError('invalid_deadline')
        self.guard = Guard(config); self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.deadline_s, self.transport_timeout_s, self.clock = deadline_s, transport_timeout_s, clock
        self.transport = transport or self._send
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='decisions-ranking')
        self.pending = None; self.count = 0

    def _record(self, event):
        with (self.log_dir/'events.jsonl').open('a') as f: f.write(json.dumps(log_value(event),allow_nan=False)+'\n')

    def _send(self, body):
        keypath = Path.home()/'.config/typhon-secrets/openai-decisions.key'
        if keypath.stat().st_mode & 0o077: raise ValueError('credential_permissions')
        key = keypath.read_text().strip()
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        # No retry on the control path. A late answer cannot change the active plan.
        with urllib.request.urlopen(req,timeout=self.transport_timeout_s) as response: return json.load(response)

    def _rank(self, observation, candidates, index):
        begin = self.clock(); folder = self.log_dir/f'{index:05d}'; folder.mkdir()
        try:
            body, panel = payload(observation, candidates); panel.save(folder/'candidates.jpg',quality=90)
            (folder/'request.json').write_text(json.dumps(body))
            start = self.clock(); raw = self.transport(body); end = self.clock()
            (folder/'response.json').write_text(json.dumps(raw,indent=2))
            result = parse(raw,list(candidates)); result.update(api_seconds=end-start,preview_seconds=start-begin,usage=raw.get('usage'))
        except Exception as exc:
            # Do not serialize exception messages or headers: they may contain secrets.
            result = {'status':'error','choice':'UNKNOWN','error_type':type(exc).__name__}
        result['worker_seconds'] = self.clock()-begin
        (folder/'result.json').write_text(json.dumps(result,indent=2))
        return result

    def submit(self, observation, candidates, context, observed_at):
        if self.pending is not None: return False
        if not math.isfinite(observed_at) or not 0 <= self.clock()-observed_at < self.deadline_s: return False
        accepted = {}; rejected = {}
        for label, values in candidates.items():
            if not isinstance(label,str) or not label.isalnum() or len(label)>8 or label=='UNKNOWN': raise ValueError('invalid_candidate_label')
            a = np.asarray(values,float).copy(); reason = self.guard.candidate(a,observation)
            if reason: rejected[label]=reason
            else: accepted[label]=a
        if not 2 <= len(accepted) <= 8:
            self._record({'type':'not_submitted','reason':'fewer_than_two_or_too_many_admissible','rejected':rejected,'accepted':list(accepted)})
            return False
        import copy
        obs = copy.deepcopy(observation); index = self.count; self.count += 1
        hashes = {k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in accepted.items()}
        self.pending = {'context':str(context),'index':index,'created':observed_at,'deadline':observed_at+self.deadline_s,
                        'candidates':accepted,'future':self.pool.submit(self._rank,obs,accepted,index),'notified':False}
        self._record({'type':'submitted','index':index,'context':str(context),'candidate_sha256':hashes,'rejected':rejected,
                      'observed_at':observed_at,'deadline':observed_at+self.deadline_s})
        return True

    def poll(self, context, observation, elapsed_steps=0, last_close=None):
        p = self.pending
        if p is None: return None
        now = self.clock()
        reason = 'context_changed' if str(context)!=p['context'] else ('deadline' if now>=p['deadline'] else None)
        if reason and not p['notified']:
            self._record({'type':'discarded','index':p['index'],'reason':reason,'age_s':now-p['created']}); p['notified']=True
        if not p['future'].done(): return None
        result = p['future'].result(); self.pending=None
        if reason or p['notified']: return None
        label = result.get('choice'); actions = p['candidates'].get(label)
        if result['status']!='valid' or actions is None:
            self._record({'type':'no_selection','index':p['index'],'result':result}); return None
        if type(elapsed_steps) is not int or not 0 <= elapsed_steps < len(actions):
            self._record({'type':'discarded','index':p['index'],'reason':'expired_action_horizon'}); return None
        if elapsed_steps and (last_close is None or np.any((actions[:elapsed_steps,9]>.5) != bool(last_close))):
            self._record({'type':'discarded','index':p['index'],'reason':'skipped_gripper_phase'}); return None
        actions = actions[elapsed_steps:]
        violation = self.guard.candidate(actions,observation)
        if violation:
            self._record({'type':'discarded','index':p['index'],'reason':violation}); return None
        self._record({'type':'selected','index':p['index'],'context':p['context'],'candidate':label,'age_s':now-p['created'],'elapsed_steps':elapsed_steps,'result':result})
        return {'candidate':label,'actions':actions.copy(),'context':p['context'],'index':p['index'],'result':result}

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
