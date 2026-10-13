"""Independent numerical/trace checks of VIDEO10 saved episodes; no environment or actor imports."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def main(out,m0,dest):
    cfg=json.loads((out/'config.json').read_text());names=['SEQ','END','SHUFFLE','STATIC','SEQ_REPEAT'];rows=[];hashes=[]
    for n in names:
        p=out/'executions'/n;r=json.loads((p/'result.json').read_text());actions=[json.loads(x) for x in gzip.open(p/'actions.jsonl.gz','rt')];final=r['final_truth'];initial=r['initial_truth'];score=r['scoring']
        xy=final['object_position_m'][:2];target=cfg['task']['target_position_xy_m'];d=math.hypot(xy[0]-target[0],xy[1]-target[1]);angle=abs(wrap(final['object_yaw_rad']-cfg['task']['target_yaw_rad']))
        assert abs(d-score['position_error_m'])<1e-12 and abs(angle-score['orientation_error_rad'])<1e-12
        yaw=final['object_yaw_rad'];c,s=math.cos(yaw),math.sin(yaw);corners=[]
        for x in [-cfg['object']['width_m']/2,cfg['object']['width_m']/2]:
            for y in [-cfg['object']['depth_m']/2,cfg['object']['depth_m']/2]:
                corners.append([xy[0]+c*x-s*y-target[0],xy[1]+s*x+c*y-target[1]])
        # Original goal yaw is0 for this frozen feeding experiment.
        assert cfg['task']['target_yaw_rad']==0.
        inside=all(abs(z)<=cfg['task']['target_region_size_m']/2+1e-9 for point in corners for z in point)
        lift=any(x['scorer_only']['bilateral_grasp_contact'] and x['scorer_only']['object_position_m'][2]-initial['object_position_m'][2]>.01 for x in actions)
        assert inside==score['footprint_inside_target'] and lift==r['supported_lift']
        assert all(x['minimum_joint_margin_rad']>=0 and not x['forbidden_static_contact'] for x in actions)
        manifest=json.loads((p/'run_manifest.json').read_text());hashes.append(tuple(manifest[k] for k in ['adapter_sha256','config_sha256','reference_sha256','protocol_sha256']))
        assert manifest['actor_truth_access'] is False and manifest['prior_expert_actions_imported'] is False
        assert sha(out/'config.json')==manifest['config_sha256'] and sha(out/'source_reference.json')==manifest['reference_sha256'] and sha(out/'protocol.json')==manifest['protocol_sha256']
        rows.append({'condition':n,'position_mm':d*1000,'yaw_deg':math.degrees(angle),'footprint_inside':inside,'supported_lift':lift,'action_count':len(actions),'minimum_joint_margin_rad':min(x['minimum_joint_margin_rad'] for x in actions)})
    assert len(set(hashes))==1
    qualified=json.loads((out/'qualified_prefix.json').read_text())['commands']
    for n in names:
        p=out/'executions'/n
        actions=[json.loads(x) for x in gzip.open(p/'actions.jsonl.gz','rt')]
        prefix=[x for x in actions if x['source_clip_time_s'] is None]
        assert [{'phase':x['phase'],'action':x['requested_action']} for x in prefix]==qualified
        assert json.loads((p/'replayed_prefix_validation.json').read_text())['aligned']
        observations=[json.loads(x) for x in gzip.open(p/'observations.jsonl.gz','rt')]
        for row in observations:
            m=row['measurement']
            if row['source_clip_time_s']>=4.9:
                assert m['release_requested'] and m['depth_hypothesis_m'] is None
                assert m['valid']==m['coordinate_conversion_valid']
                if m['valid']:assert m['view']=='primary'
                else:assert m['feature'] is None
                assert row['supplied_motion'] and 'correction' not in row
        result=json.loads((p/'result.json').read_text())
        assert result['scoring']['success']==all(result['scoring']['criteria'].values())
    exact={}
    for file in ['actions.jsonl.gz','observations.jsonl.gz']:
        exact[file]=gzip.decompress((out/'executions/SEQ'/file).read_bytes())==gzip.decompress((out/'executions/SEQ_REPEAT'/file).read_bytes())
        assert exact[file]
    verified=[]
    for src in json.loads((out/'source_provenance.json').read_text())['originals']:
        marker='Projects/humanoid-task-supervisor/'
        if marker in src['path']:
            path=m0/src['path'].split(marker,1)[1]
            assert sha(path)==src['sha256'];verified.append({'relative_path':str(path.relative_to(m0)),'sha256':src['sha256']})
    assert len(verified)==3
    result={'passed':True,'independent_physical_recalculation':rows,'same_code_config_reference_protocol_for_all_runs':True,'repeat_exact':exact,'shared_qualified_prefix_action_identical_for_all_five':True,'postrelease_independent_or_unknown':True,'verified_original_sources_and_template':verified,'scope':'independent arithmetic and file/trace checks; does not establish accuracy of the carried-depth observer'}
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--m0',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();main(a.output,a.m0,a.destination)
