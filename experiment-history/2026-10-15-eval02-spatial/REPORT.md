# EVAL02-SPATIAL: renderer isolation failed before model evaluation

The planned8-trial numeric-position × visual-marker-position experiment did not run. The renderer-only qualification stopped at the initial scene's negative20mm marker displacement: the secondary image changed29pixels inside the protected robot/gripper/block/fixture mask. The exact foreground-isolation gate failed.

This is an implementation qualification failure, not evidence that the model follows or ignores a moved marker. There are zero policy queries and zero scored rollouts.

## Intended intervention

Follow-up to EVAL02-XY#11059; ticket#11060. Same LOAD04-B, demo0002, frozen physical square, two paired seeds. Numeric requestedY and decorative markerY would independently take-.140/-.100m, yielding8trials. No training, harness, TTC, holdouts, geometry/dynamics modification or stopping-rule change.

Five noncolliding staging-border/orientation geoms temporarily receive a derived world-position Y displacement during the existing native renderer call. All geom_xpos values restore before returning. No model/mocap changes or forward/integration calls are introduced by the wrapper.

## Actual qualification

- Fresh reset matched original training truth and qpos/qvel/mocap arrays.
- Native initial camera frames matched the archived inputs.
- Native no-op images matched exactly.
- Model/physical fingerprint and all derived geom positions restored exactly after shifted renders.
- Primary changed-marker test proceeded; secondary failed with29 protected foreground pixels changed.
- Initial crop test and deliberate foreground-occlusion fixtures were not reached.
- The script raises before saving its final contact sheet or per-condition image arrays. Pixel cause cannot be diagnosed from the retained log alone; do not invent a shadow/occlusion explanation.
- No threshold changed, condition retried, alternative seed selected or renderer engineering branch reopened.

The original15min gate stopped in seconds. The broader programme remains active; next work is a bounded read-only goal/action coverage audit.

## Reproduction and preservation

Source: scripts/eval02_spatial_render.py and scripts/eval02_spatial_qualify.py. Qualification log, original clock and failure traceback are preserved beside this report and in the Mint archive. Run the qualification script with the pinned canonical simulator interpreter on this branch and existing EVAL02-XY/LOAD04 source files. The frozen output path already contains the failure and must not be overwritten. This report records a failed qualification; it is not a completed factorial model comparison.

All earlier results/checkpoints remain unchanged. EVAL02's68intervention trials remain unexecuted. AlphaNIST and observer authority remain parked/blocked.
