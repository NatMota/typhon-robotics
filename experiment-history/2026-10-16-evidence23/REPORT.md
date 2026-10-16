# EVIDENCE23 — Historical frame copies offloaded after verification

Completed work, ticket #11096. No inference, training, simulation or new result evaluation ran.

The historical AB40-v2 saved-observation frames already had a Mint archive. Freshly hashing both copies allowed removal of only the redundant numbered frame PNGs from Powerglide's local scratch disk. This operation did **not** free the user's persistent HOME quota.

| Measure | Observed result |
|---|---:|
| Removed redundant numbered frame copies | 64,784 |
| Logical frame bytes retained on Mint | 17,812,317,101 |
| Remaining original files rehashed after offload | 5,570 |
| Remaining-file errors | 0 |
| Local filesystem free bytes before | 485,691,392 |
| Local filesystem free bytes after | 18,432,172,032 |

Source root: `/var/tmp/typhon-ab40-30s-parallel-20261010-v2`.

Retained archive root: `/media/nm/Fast Drive/Typhon/archives/powerglide-full-20261014/var-tmp/typhon-ab40-30s-parallel-20261010-v2`.

Selection was limited to numbered `[0-9]+-(primary|secondary).png` frames. Initial, final and goal images, videos, arrays, logs, models and lab files were excluded. All original files remaining in the source root were rehashed against their saved pre-offload hashes; the subsequent location note is an explicit new file.

The initial selector rejected an unexpected goal-image basename before deletion. The corrected selector explicitly retained goal images. Its earlier failed attempt remains in the full Mint evidence directory.

## Verification chain

1. Selected Mint files matched the reconciled full-backup SHA-256 ledger.
2. Exact Powerglide counterparts matched the verified Mint manifest.
3. Verification metadata was archived on Mint before source removal.
4. Every selected Powerglide file was rehashed immediately before unlinking.
5. All 5,570 remaining original files passed the after-offload check.
6. The final evidence directory was copied back and verified: 16 files, 57,329,532 bytes, zero errors.

The final archive is `/media/nm/Fast Drive/Typhon/research-20261015/evidence23-20261016/powerglide-final`. Its manifest SHA-256 is `e46e5840294e52e1022a80eb4b749a5737a805745453329cb8dbdd839103c92d`. The full per-file manifests remain there rather than being duplicated in Git.

Copied receipts in this report are byte-identical to that archive. `copied-receipts-manifest.json` records their hashes. Reproduction is an integrity audit of the retained evidence; no deletion operation should be replayed.

## Boundaries

This is evidence preservation and local-storage recovery, not model improvement or a new scientific result. Original experiment scores and checkpoints were not revised. The later HOME-quota repair is separate work, EVIDENCE24.
