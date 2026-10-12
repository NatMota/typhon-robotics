# Storage and experiment branches

GitHub repository: https://github.com/NatMota/typhon-robotics (private).

| Content | Durable location |
|---|---|
| Code, configs, protocols, small manifests, result tables and reports | GitHub experiment branches |
| Model/optimizer checkpoints, full video, raw predictions, arrays and rollout traces | Mint Fast Drive |
| Working environments and active simulation/training jobs | Powerglide; reconstruct from source/configs |

A Git worktree is a working directory for a branch. Push the branch to GitHub; the worktree itself is not a separate remote artifact. The three VIDEO01 worktrees start from the same saved research commit and can run independently without overwriting one another.

Current bulk source: `/var/tmp/typhon-ost-sweep-20261012` on Powerglide.
Backup destination: `/media/nm/Fast Drive/Typhon/research-20261012/ost-sweep-20261012` on Mint.
VIDEO01 output source: `/var/tmp/typhon-video-representation-20261012`; corresponding Mint destination is `/media/nm/Fast Drive/Typhon/research-20261012/video01`.
Frozen M0 source: `~/Projects/humanoid-task-supervisor/data/personal/m0` on Powerglide. Raw source files and prior results stay unchanged.

A backup is complete only when a receipt confirms transferred file counts and SHA-256 verification. A destination path or an active rsync does not prove completion. Check the newest receipt in `experiment-history/2026-10-12-ost-sweep/backup/` and on Mint. Original Powerglide files are retained until verified.

## Restore

1. Clone the private repository and check out the required experiment branch.
2. Read its `configs/experiments/video01/*.json` and the shared VIDEO01 protocol.
3. Restore the relevant frozen data/checkpoints from Mint, verifying the SHA-256 manifest. For older experiments, use their original manifests and model identities; do not substitute another checkpoint of the same milestone name.
4. Recreate the documented Python environment, configure read-only paths to data and model caches, and run into a new output directory.

Do not put credentials, environments, model files or bulk media into Git. `.gitignore` excludes common forms; explicit file selection is still required. GitHub rejects regular Git files over 100 MiB; LFS is a separate storage service with per-file and billing limits. We are not enabling paid LFS here.

## Verified 12 October backup

The new OST sweep has 4,907 files (29,566,551,543 bytes) verified byte-for-byte on Mint, including all three new checkpoints. VIDEO01 has 622 verified input files (137,240,303 bytes), preserving raw/curated/excluded M0 media, plus 25 verified extraction/code artifacts (27,958,500 bytes). The code archive passed a bare-clone restore and Git fsck. Read the receipts and per-file manifests in `experiment-history/2026-10-12-ost-sweep/backup/`. No Powerglide source files were removed.

Powerglide reservation AAE1BB6A was checked against the live UCL service on 12 October: scheduled end is Tuesday 13 October 2026 at 01:00 BST. This is the access deadline, not a guarantee about file retention.
