# INIT02 — matched transition supervision

Status: protocol and inputs qualified; no new model has trained at this freeze.

P is unchanged FL. R and T start from the identical FL full weights with fresh AdamW, seed151901, exactly3000 updates and effectivebatch4. Both receive identical4500 priorfeeding,4500 priorloading and1500 preservation presentations. The remaining1500 are oldloading in R and newINIT01 loading suffixes in T. These are expert-generated robot actions at four closely spaced terminal states, not measured human motor commands.

69 fixed native-policy episodes,23/model:8 initiation,8 preservation,3 priorfeeding,4 priorloading. H30,O2,domain29,10 refinements,1436 actions+24settle. No harness,TTC,teacher,router or completion stop. Two evaluation workers maximum; training runs alone. Optional12 quarantined development episodes only under the exact predeclared joint gate. See PLANNER_PROTOCOL.md.

## Commands

From this private worktree, with original datasets and the documented environments available:

```sh
.venv-sim/bin/python scripts/init02_prepare.py freeze
.venv-sim/bin/python scripts/init02_prepare.py prepare
.venv-xvla/bin/python scripts/init02_train.py gate
# Commit, verify private push, and seal execution-freeze.json before GPU work.
.venv-sim/bin/python scripts/init02_eval.py --model P --lane 0
.venv-sim/bin/python scripts/init02_eval.py --model P --lane 1
.venv-xvla/bin/python scripts/init02_train.py train --variant R
.venv-sim/bin/python scripts/init02_eval.py --model R --lane 0
.venv-sim/bin/python scripts/init02_eval.py --model R --lane 1
# SHA-verify R checkpoint on Mint and offload local copy before T.
.venv-xvla/bin/python scripts/init02_train.py train --variant T
.venv-sim/bin/python scripts/init02_eval.py --model T --lane 0
.venv-sim/bin/python scripts/init02_eval.py --model T --lane 1
```

Paths are frozen to /var/tmp/typhon-init02-20261015. Reproduction must use a fresh output namespace; original outputs are never overwritten. Optimizer is not saved: a fresh full-recipe rerun from FL is required for interrupted training. Phase deadlines remain fixed review gates.
