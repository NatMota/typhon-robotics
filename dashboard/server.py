#!/usr/bin/env python3
"""Read-only M7 evidence console. Python standard library; loopback only."""
import argparse
import gzip
import json
import mimetypes
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

HERE = Path(__file__).resolve().parent
EVIDENCE = Path(os.environ.get('M7_EVIDENCE_DIR', '/var/tmp/humanoid-m7-live-evaluation-20261009'))
ARCHIVE = Path(os.environ.get('M7_TRAINING_DIR', '/var/tmp/humanoid-runpod-archive-20261009'))
LIVE = Path(os.environ.get('M7_LIVE_DIR', str(EVIDENCE / 'live')))
SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def within(root, *parts):
    if any(not SAFE.fullmatch(p) or p in ('.', '..') for p in parts):
        raise ValueError('Invalid evidence identifier')
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Invalid path')
    return path

def counts(results, completed):
    rows = [r for r in results if (r.get('group') == 'completed') == completed]
    return {'passed': sum(r.get('success') is True for r in rows), 'total': len(rows)}

def training_jobs(tree, timestamp):
    """Watchers may publish flat jobs or nested run-number groups."""
    jobs = []
    if not isinstance(tree, dict):
        return jobs
    for name, value in tree.items():
        if not isinstance(value, dict):
            continue
        if 'state' in value or 'updates_seen' in value:
            target = value.get('expected_updates')
            if target is None and re.fullmatch(r'v[123]-m[3456]', name):
                target = 1000 if name.endswith(('m3', 'm4')) else 3000
            jobs.append(dict(value, id=name, updated_epoch=timestamp, target=target))
        else:
            jobs.extend(training_jobs(value, timestamp))
    return jobs


def state():
    now = time.time()
    runs = []
    for path in sorted(EVIDENCE.glob('*/summary.json')):
        summary = read_json(path)
        if not isinstance(summary, dict) or not summary.get('results'):
            continue
        cp = read_json(path.parent / 'checkpoint_load.json')
        results = summary['results']
        runs.append({'id': path.parent.name, 'label': path.parent.name.replace('-pilot-v1', '').upper(),
                     'updated_epoch': path.stat().st_mtime, 'results': results,
                     'manipulation': counts(results, False), 'preservation': counts(results, True),
                     'checkpoint': {k: cp.get(k) for k in ('ready', 'step', 'schema')},
                     'evidence_type': summary.get('evidence_type', 'learned-policy rollout')})
    training = []
    watcher = read_json(ARCHIVE / 'watcher/watch-status.json')
    for pod in watcher.get('pods', {}).values():
        training.extend(training_jobs(pod.get('jobs', {}), pod.get('last_checked_epoch')))
    watcher3 = read_json(ARCHIVE / 'run3-watcher/status.json')
    training.extend(training_jobs(watcher3.get('jobs', {}), watcher3.get('heartbeat_epoch')))
    live = read_json(LIVE / 'frame.json')
    if live:
        stamp = live.get('timestamp_unix', live.get('timestamp_epoch', live.get('timestamp', (LIVE / 'frame.json').stat().st_mtime)))
        if not isinstance(stamp, (float, int)):
            stamp = (LIVE / 'frame.json').stat().st_mtime
        live['age_s'] = max(0, now - stamp)
        live['fresh'] = live['age_s'] < 15 and live.get('live', True) is not False
        live['available'] = (LIVE / 'latest.jpg').is_file()
    return {'server_epoch': now, 'runs': runs, 'training': sorted(training, key=lambda j: j['id']),
            'run4_status': watcher3.get('hold', {}).get('state') or watcher3.get('phase', 'pending'), 'live': live,
            'observer_only': True, 'schema': 'm7-observer-v1'}

def episode(run, case):
    folder = within(EVIDENCE, run, case)
    data = read_json(folder / 'episode.json')
    if not data:
        raise FileNotFoundError('Episode not available')
    scenario = read_json(folder / 'scenario.json')
    keys = ['scenario_id', 'group', 'task', 'seed', 'started_at_utc', 'provenance', 'success', 'status',
            'initial_truth', 'final_truth', 'position_error_mm', 'yaw_error_deg', 'grasp_attempts',
            'failed_grasps', 'recovery_attempts', 'recovery_success', 'controller_exceptions',
            'episode_steps', 'total_control_steps', 'simulated_duration_s', 'wall_duration_s',
            'mean_inference_latency_s', 'policy_actions_executed', 'artifact_errors', 'evaluation_errors',
            'inference_pauses_simulation', 'move_events']
    result = {k: data.get(k) for k in keys}
    result['evaluation'] = {k: v for k, v in data.get('evaluation', {}).items() if k != 'attempts'}
    result['attempts'] = data.get('attempts', [])
    result['criteria'] = data.get('evaluation', {}).get('criteria', {})
    result['limits'] = scenario.get('config', {}).get('evaluation', {})
    result['media'] = [n for n in ('rollout.mp4', 'initial.png', 'final.png') if (folder / n).is_file()]
    return result

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print('%s %s' % (self.log_date_time_string(), fmt % args), flush=True)

    def json(self, obj, status=200):
        body = json.dumps(obj, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def file(self, path):
        if not path.is_file():
            raise FileNotFoundError('File not available')
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get('Range', '')
        if requested:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
            if not match:
                return self.json({'error': 'Unsupported range'}, 416)
            start = int(match[1])
            end = min(int(match[2]) if match[2] else size - 1, size - 1)
            if start > end:
                return self.json({'error': 'Range outside file'}, 416)
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(262144, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == '/api/state':
                return self.json(state())
            if url.path in ('/api/episode', '/api/trajectory'):
                run, case = query.get('run', [''])[0], query.get('case', [''])[0]
                if url.path == '/api/episode':
                    return self.json(episode(run, case))
                path = within(EVIDENCE, run, case, 'trajectory.jsonl.gz')
                with gzip.open(path, 'rt') as stream:
                    return self.json([json.loads(line) for line in stream])
            if url.path == '/live/latest.jpg':
                return self.file(LIVE / 'latest.jpg')
            if url.path.startswith('/media/'):
                parts = unquote(url.path).split('/')[2:]
                if len(parts) != 3 or parts[-1] not in ('rollout.mp4', 'initial.png', 'final.png', 'episode.json', 'scenario.json', 'trajectory.jsonl.gz'):
                    raise FileNotFoundError('Unknown evidence file')
                return self.file(within(EVIDENCE, *parts))
            if url.path in ('/', '/index.html', '/app.js', '/styles.css'):
                return self.file(HERE / ('index.html' if url.path == '/' else url.path[1:]))
            raise FileNotFoundError('Unknown endpoint')
        except (BrokenPipeError, ConnectionResetError):
            pass
        except FileNotFoundError as exc:
            self.json({'error': str(exc)}, 404)
        except (ValueError, KeyError, OSError) as exc:
            self.json({'error': str(exc)}, 400)

def self_check():
    assert counts([{'group': 'completed', 'success': True}, {'group': 'source', 'success': False}], False) == {'passed': 0, 'total': 1}
    assert counts([{'group': 'completed', 'success': True}], True) == {'passed': 1, 'total': 1}
    for bad in ('..', '../secret', 'a/b', ''):
        try:
            within(EVIDENCE, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe path accepted')
    nested = training_jobs({'3': {'v3-m3': {'state': 'complete', 'updates_seen': 1000}}, '4': {'v4-m4': {'state': 'complete', 'updates_seen': 520, 'expected_updates': 520}, 'v4-m5': {'state': 'pending', 'updates_seen': None, 'expected_updates': None}}}, 1)
    assert [(j['id'], j['target']) for j in nested] == [('v3-m3', 1000), ('v4-m4', 520), ('v4-m5', None)]
    snapshot = state()
    assert snapshot['observer_only'] is True
    for run in snapshot['runs']:
        assert run['manipulation']['total'] + run['preservation']['total'] == len(run['results'])
    print('PASS: outcome separation, path boundaries, nested watcher targets, current evidence inventory')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        self_check()
    else:
        print(f'M7 observer listening on http://127.0.0.1:{args.port}', flush=True)
        ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
