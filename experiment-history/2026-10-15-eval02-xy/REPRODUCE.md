# Reproduce EVAL02-XY

Use this branch on Powerglide with the pinned environments and sources in protocol.json. Select a fresh output root to preserve all original artifacts.

~~~
export TYPHON_EVAL02_XY_ROOT=/var/tmp/typhon-eval02-xy-reproduction
mkdir "$TYPHON_EVAL02_XY_ROOT"
python3 - <<'PY'
import os,json,time,pathlib
r=pathlib.Path(os.environ['TYPHON_EVAL02_XY_ROOT']);n=time.time()
(r/'clock.json').write_text(json.dumps(dict(start_epoch=n,preparation_deadline=n+1200,execution_deadline=n+1800,review_deadline=n+2700)))
PY
.venv-xvla/bin/python scripts/eval02_xy.py prepare
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/eval02_xy.py run
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/eval02_xy.py analyse
~~~

Preparation recomputes fixed-square scoring checks, model/source hashes, exact centre wording and processor token preservation. Both centre controls must pass before offsets. Technical qualification failures stop the panel; no replacement seed/scene. New clock defines a new reproduction, not revision of the frozen original.

Existing dependencies: /var/tmp/typhon-eval02-goal-20261015, /var/tmp/typhon-live01-p-20261015, /var/tmp/typhon-load04-20261014. Symlink .venv-xvla and .cache-xvla to installed paths used by the previous experiment. Preserve the canonical simulator interpreter path, because asset resolution depends on it. No upgrades.

Original runtime: /var/tmp/typhon-eval02-xy-20261015.
Mint archive: /media/nm/Fast Drive/Typhon/research-20261015/eval02-xy-steering.
Native movies: runtime/episodes/*/rollout.mp4.
Three-target comparisons: runtime/comparison-videos/.

Optional read-only extra checks: run scripts/eval02_xy_review.py with the same output root after analyse.
