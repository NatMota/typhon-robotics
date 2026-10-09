#!/usr/bin/env python3
"""Small integration check against a running, populated observer console."""
import json
import sys
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

base = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8765').rstrip('/')

def get(path):
    with urlopen(base + path, timeout=10) as response:
        return json.load(response)

state = get('/api/state')
assert state['observer_only'] is True
assert state['runs'], 'This integration check needs recorded evaluation evidence'
for run in state['runs']:
    for field, completed in [('manipulation', False), ('preservation', True)]:
        rows = [r for r in run['results'] if (r.get('group') == 'completed') == completed]
        assert run[field] == {'passed': sum(r.get('success') is True for r in rows), 'total': len(rows)}
run = state['runs'][0]
case = run['results'][0]['scenario_id']
query = urlencode({'run': run['id'], 'case': case})
episode = get('/api/episode?' + query)
trajectory = get('/api/trajectory?' + query)
assert episode['scenario_id'] == case and trajectory
assert trajectory[0]['step'] <= trajectory[-1]['step']
video = next(((r['id'], e['scenario_id']) for r in state['runs'] for e in r['results'] if e.get('video_saved')), None)
if video:
    path = '/media/' + '/'.join(quote(p, safe='') for p in video) + '/rollout.mp4'
    with urlopen(Request(base + path, headers={'Range': 'bytes=0-99'}), timeout=10) as response:
        assert response.status == 206 and len(response.read()) == 100
        assert response.headers['Content-Range'].startswith('bytes 0-99/')
for request in [Request(base + '/media/%2e%2e/secret/episode.json'), Request(base + '/api/state', method='POST')]:
    try:
        urlopen(request, timeout=10)
    except HTTPError as error:
        assert error.code in (400, 404, 501)
    else:
        raise AssertionError('Unsafe path or write method unexpectedly accepted')
live = state.get('live', {})
if live.get('live') is False:
    assert live['fresh'] is False, 'Finished observer falsely labelled live'
print(json.dumps({'result': 'PASS', 'runs': len(state['runs']),
                  'episodes': sum(len(r['results']) for r in state['runs']),
                  'training_jobs': len(state['training']), 'video_range_checked': bool(video),
                  'checks': ['separate outcome groups', 'episode and trajectory', 'MP4 byte range',
                             'path traversal rejection', 'read-only methods', 'finished observer status']}, indent=2))
