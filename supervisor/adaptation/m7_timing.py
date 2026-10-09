"""Execute explicit requested anchors on the original20Hz clock; no future-label inference."""
import math
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
HZ=20
def _tick(value):
    if not math.isfinite(value) or abs(value*HZ-round(value*HZ))>1e-7:
        raise ValueError('Timing must fall exactly on original20Hz control ticks')
    return int(round(value*HZ))
def requested_grid(duration_s,slots=30):
    n=_tick(duration_s)
    if n<1 or not isinstance(slots,int) or not 1<=slots<=30:raise ValueError('Positive duration and1..30slots required')
    return [int(i)/HZ for i in np.rint(np.linspace(0,n-1,min(slots,n))).astype(int)]
def _source(values):
    result=np.asarray(values,dtype=float).copy()
    for i in (3,13):result[i:i+6]=result[i+np.array([0,2,4,1,3,5])]
    return result
def _generic(values):
    result=np.asarray(values,dtype=float).copy()
    for i in (3,13):result[i:i+6]=result[i+np.array([0,3,1,4,2,5])]
    return result
def _matrix(rot6):
    a,b=np.asarray(rot6[:3]),np.asarray(rot6[3:])
    if np.linalg.norm(a)<1e-8:raise ValueError('Degenerate predicted rotation firstcolumn')
    a=a/np.linalg.norm(a);b=b-a*np.dot(a,b)
    if np.linalg.norm(b)<1e-8:raise ValueError('Degenerate predicted rotation secondcolumn')
    b=b/np.linalg.norm(b);return np.column_stack([a,b,np.cross(a,b)])
def reconstruct_20hz(payload,oracle_replay=False):
    request=payload['request'];source=request['source']
    if source not in ['user','planner','policy_prediction','training_demonstration_supervision'] or (source=='training_demonstration_supervision' and not oracle_replay):
        raise ValueError('Runtime requires user/planner/policy timing; future demonstration labels forbidden')
    if request['scope'] not in ['single','pair','triple','intent'] or not request['instruction']:raise ValueError('Explicit requested scope/instruction required')
    n=_tick(float(request['aligned_duration_s']));points=payload['waypoints']
    times=[float(p['time_s']) for p in points]
    if times!=list(request['anchor_times_s']) or not 1<=len(times)<=30:raise ValueError('Waypoints differ from requested anchor timing')
    ticks=np.array([_tick(t) for t in times])
    if n<1 or ticks[0]!=0 or ticks[-1]>=n or np.any(np.diff(ticks)<=0):raise ValueError('Invalid ordered control ticks')
    anchors=np.asarray([_source(p['target_generic_interleaved']) for p in points])
    if anchors.shape!=(len(points),20) or not np.isfinite(anchors).all():raise ValueError('Expected finite20D targets')
    grid=np.arange(n);targets=np.column_stack([np.interp(grid,ticks,anchors[:,c]) for c in range(20)])
    for start in (3,13):
        rot=anchors[:,start:start+6]
        if np.all(rot==0):targets[:,start:start+6]=0;continue
        matrices=np.asarray([_matrix(r) for r in rot])
        if len(ticks)>1:
            values=Slerp(ticks,Rotation.from_matrix(matrices))(np.minimum(grid,ticks[-1])).as_matrix()
            targets[:,start:start+6]=np.concatenate([values[:,:,0],values[:,:,1]],axis=1)
        else:targets[:,start:start+6]=rot[0]
    # Native target values retained at exact anchors; grippers arebinary decisions with zero-orderhold.
    targets[ticks]=anchors
    last=np.searchsorted(ticks,grid,side='right')-1
    for c in (9,19):targets[:,c]=(anchors[last,c]>.5).astype(float)
    events=[{'control_index':int(i),'time_s':i/HZ,'channel':c,'value':int(targets[i,c])} for c in (9,19) for i in range(1,n) if targets[i,c]!=targets[i-1,c]]
    return {'targets':targets,'timestamps_s':grid/HZ,'gripper_events':events,'request':request,
            'rotation_encoding':'concatenated_columns','control_hz':HZ,
            'final_target_source':targets[-1].copy(),'final_target_generic_interleaved':_generic(targets[-1]),
            'reconstruction':'Cartesianlinear, SO3SLERP betweenanchors, lastposehelduntil duration, binarygripperzero-orderhold; exactsourcecontrolticks'}
def execute_timed(env,payload,oracle_replay=False):
    from .actions import robot_observation
    from supervisor.baseline.xvla import to_action
    result=reconstruct_20hz(payload,oracle_replay)
    for target in result['targets']:
        env.step(to_action(target,robot_observation(env),env.config))
    return result

