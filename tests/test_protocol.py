import numpy as np
from supervisor.core import Observer,GOALS,draw_board,rule_action,WAIT,FINISH,UNAVAILABLE
from supervisor.env import TaskEnv,MODES


def test_occlusion_is_uncertainty_not_absence_or_completion():
    observer=Observer();req=np.array([True,False,False,False]);positions=GOALS.copy();positions[0]=[.2,.2]
    h,m=observer.observe(draw_board(positions),req)
    old=h[-1,0,:2].copy()
    positions[0]=GOALS[0]
    h,m=observer.observe(draw_board(positions,hidden=[0]),req)
    assert not m[0] and h[-1,0,5]==0 and h[-1,0,6]==0
    assert np.array_equal(h[-1,0,:2],old)
    h,m=observer.observe(draw_board(positions),req)
    assert rule_action(h,m)==FINISH
    assert m[0]  # Completed parts stay feasible; the learned method gets no skip rule.


def test_stop_is_shared_and_mask_allows_only_wait():
    e=TaskEnv();h,m=e.reset(goal_size=2,mode='protected',seed=99)
    (h,m),_,_,_=e.step(WAIT)
    assert np.flatnonzero(m).tolist()==[WAIT]
    before=e.positions.copy();(h,m),_,_,_=e.step(0)
    assert e.metrics['unsafe_continuation']==1 and np.array_equal(before,e.positions)


def test_completion_requires_truth_not_actor_finish_claim():
    e=TaskEnv();e.reset(seed=1);_,_,done,r=e.step(FINISH)
    assert done and not r['success'] and r['false_completion']==1


def test_reference_all_cases_without_false_success():
    for mode in ['clean']+MODES:
        e=TaskEnv();h,m=e.reset(goal_size=4,mode=mode,seed=101)
        for _ in range(32):
            (h,m),_,done,r=e.step(rule_action(h,m))
            if done:break
        assert r['reason']==('required_part_unavailable' if mode=='unavailable' else 'completed'), (mode,r)
        assert r['false_completion']==0 and r['duplicate']==0 and r['unsafe_continuation']==0
        if mode=='grasp_failure':assert r['grasp_failures']==1 and r['retries']==1


def test_observer_independent_of_evaluator_truth():
    a,b=TaskEnv(),TaskEnv();a.reset(seed=7);b.reset(seed=7)
    image,_=a._camera();b.positions[:]=GOALS
    # Same RGB / supplied goals / history must produce identical actor inputs.
    x=Observer().observe(image,a.required)
    y=Observer().observe(image,b.required)
    assert np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1])


def test_whole_episode_and_task_holdout():
    import json
    from pathlib import Path
    p=Path('data/processed/manifest.json')
    if not p.exists():return
    m=json.loads(p.read_text());sets=[set(x) for x in m['splits'].values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    for condition in ['clean','disturbed']:
        d=np.load(f'data/processed/train_{condition}.npz')
        assert set(d['episode'])<=sets[0]
        assert np.all(d['x'][:,-1,:,4].sum(-1)<=2)
        assert np.all(d['mask'][np.arange(len(d['y'])),d['y']])


def test_calibrated_blue_under_bright_light():
    rgb=np.full((128,128,3),70,np.uint8);rgb[40:47,60:67]=[140,193,255]
    h,m=Observer().observe(rgb,np.array([0,1,0,0],bool))
    assert m[1] and h[-1,1,5]==1 and not m[0]
