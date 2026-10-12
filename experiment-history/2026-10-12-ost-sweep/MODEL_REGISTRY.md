# Model and data registry

The milestone names are experiment stages, not one sequential chain of weights. In particular, V4-M6 continues V3-M6; broad-Pt12k is a separate fresh Pt adaptation. New fixed/varied/recovery models all continue that same broad-Pt12k.

## Verified checkpoint headers

| Checkpoint | Updates in saved stage | Initial checkpoint | Mode / observation | SHA-256 |
|---|---:|---|---|---|
| v2-m3.pt | 1000 | fresh pinned base / see original manifest | m3_prompt / dense | 2b29aa0e5d7c2e950ab7e71117038b79e6e06a3411aab8c65dec8a78ddcf401b |
| v2-m4.pt | 1000 | /workspace/humanoid-m7/results/personal-m7-training-20261009/expert-pair-20261009/runs/v2-m3/m7-step001000.pt | m4_recovery / dense | 58ed30f98262a8749c1ca94190bc3d9da533fdcb15e114b16f44306f26f1f8b6 |
| v2-m5.pt | 3000 | fresh pinned base / see original manifest | m5_lora / dense | 8ffa891c7ab1da55b3d3a97d1d45a4d8e197cab4448d0af5309fbb95e34c1f2b |
| v3-m3.pt | 1000 | fresh pinned base / see original manifest | m3_prompt / sparse | 2e40239ea5e1f49d587c92329cd317965ce8f7bf753035c1a727580b32ef4ed1 |
| v3-m4.pt | 1000 | /workspace/humanoid-m7/results/personal-m7-training-20261009/temporal-run3-20261009-v2/runs/v3-m3/m7-step001000.pt | m4_recovery / sparse | 0adb1dfff00289952013fcdeb026c9ac730f6bdcfd24c67b10e4b0b2ab7aab15 |
| v3-m5.pt | 3000 | fresh pinned base / see original manifest | m5_lora / sparse | 49d1e790c2a23c219cd3088628a7270f0983687d8fd3275f194a9ba3426ccc51 |
| v3-m6.pt | 3000 | fresh pinned base / see original manifest | m6_joint / sparse | af87b969d15f646d2d5ba497369e473e06cbd4c7f13b3591934c5910e842b1ed |
| v4-m3.pt | 500 | /workspace/humanoid-m7/results/personal-m7-training-20261009/temporal-run3-20261009-v2/runs/v3-m3/m7-step001000.pt | m3_prompt / sparse | 8cc2fb93dc2a8cb0ea51364febfa22dc453577a1b0ae1932ecd05ad47229f33b |
| v4-m4.pt | 520 | /workspace/humanoid-m7/results/personal-m7-training-20261009/temporal-run3-20261009-v2/runs/v3-m4/m7-step001000.pt | m4_recovery / sparse | 7d80bc0fbbb13c2f6c21b6871f00d1e4919ea96707b656b6eb0b2536c7f411a5 |
| v4-m6.pt | 500 | /workspace/humanoid-m7/results/personal-m7-training-20261009/temporal-run3-20261009-v2/runs/v3-m6/m7-step003000.pt | m6_joint / sparse | c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc |
| personal-m3-20261009/training/P17/prompt.pt | 1000 | see original report; no inferred lineage | O2 / current views | de63f0f46aa41913eac33207a287e24e15f241deeaef88f27b9dbde59fd00964 |
| personal-m3-20261009/training/P29/prompt.pt | 1000 | see original report; no inferred lineage | O2 / current views | cbfe48bd67d6a7806602815f1eef50d0f84e877d4e9643ead194323a050fa751 |
| personal-m3-20261009/training/S17/prompt.pt | 1000 | see original report; no inferred lineage | O2 / current views | c4b60c9bce71270ea1419d6711ffd2e8527db2e6a7abdfc9bc48714aeac11661 |
| personal-m3-20261009/training/S29/prompt.pt | 1000 | see original report; no inferred lineage | O2 / current views | 2361f14b4fd9f2f20c284c30263607d6ba09b3536368d00586e048d4fdbf2c28 |
| personal-m4-20261009/training/PN/prompt.pt | 500 | see original report; no inferred lineage | O2 / current views | a242d346e8a858ffa949ff8dd811090ab38cdd540723a94f46150f7f19ece747 |
| personal-m4-20261009/training/PR/prompt.pt | 500 | see original report; no inferred lineage | O2 / current views | 8d2f726571ecbe7dc3676232cec8e520f22a4c6fe080df494da26293d4a46ad5 |
| final.pt | 3000 | see original report; no inferred lineage | O2 / current views | 03a55fa46088368da0b8d92091ec3a4a5a539ae2772bf2cfd4c592b92b745a85 |
| step3000.pt | 3000 | fresh pinned base / see original manifest | O2 / current views | 970bb00faaa19c80715df5a3b37c827dda2300efedcb6d49f7d26069d27de4a1 |
| step12000.pt | 12000 | fresh pinned base / see original manifest | O2 / current views | 59eb9e502c057d639e278e7f84d06c8fa8e3e513e52522dae47089a97b5c3736 |

## M6 comparison identities

V1/V2-M6 identity below is from the frozen775 suite manifest; a current checkpoint-header reread was not included in the19-header audit. Do not label that as new header verification.

- v1-m6: `fc80f1d8af510ad1b70cf8e0969065567e1f67e3e415d2f27fcbf7f7c54f7986`
- v2-m6: `87b68bc9ff67c737957e8a8b5c334514cca1a36302ba75a5eabf11a326001b56`
- v3-m6: `af87b969d15f646d2d5ba497369e473e06cbd4c7f13b3591934c5910e842b1ed`
- v4-m6: `c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc`

## New continuations

| Model | Same initial weights | Updates | Data | Final SHA-256 |
|---|---|---:|---|---|
| fixed | broad-Pt12k | 3000 joint, fresh optimizer | six original-goal teacher trajectories | 007892426a061fe903e188d41f26b7432830becdfd583435e3373df728029fdd |
| varied | broad-Pt12k | 3000 joint, fresh optimizer | 12trajectories, two goals per each of six sources; six third combinations withheld | d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb |
| recovery | broad-Pt12k | 3000 joint, fresh optimizer | six nominal +four qualified learner-state corrections from two source families | 5f3e4dde3e664d0a384a58742262380164902aafea75ee7560fa5b091c0870e4 |

Native broad-Pt and its new continuations: current paired RGB plus detected RGB crop (O2), supplied task text, 20D native proprio/action interface; absolute end-effector XYZ, 6D rotation, gripper;30future commands at20Hz=1.5s. No four-second history branch or privileged object-state policy input. Full per-update sample identities retained.

V2/V3/V4 temporal experiments: generated robot RGB/action data with dense/sparse causal history, requested scope/time conditioning and declared privileged held annotations. Do not describe these as pure RGB deployment. Unknown held annotations are masked, not invented negatives.

V2 candidate catalogue:632 rows=316semantic chunks x2view orders. V3:776/388. Chunks span several durations; four-second observation history does not imply every future target is four seconds. Full aggregate source coverage verified for V1/V2; equivalent historical per-update IDs absent. See audits/training-data-audit.json.

Historical phone held exports:16takes,32camera videos,7904view-frames;1024held,3598not-held,3282unknown;768green-on frames. Those are historical16 before later evaluation exclusion, not16validated direct Franka-action videos. Unknown/off must not be used as a negative held label.

Geometry: preserved training proxy50x35x30mm,80g. Catalogue surrogate32x43x32mm,12.7g is different and not a measurement of the user piece. Table mapping uses a virtual100mm target, not a measured physical scale. Keep both declared until calibration justifies a new version.

Six older M3/M4 files were SHA-verified on Mint Fast Drive before offloading their PG working copies. Original reports/configurations remain unchanged. Registry paths can therefore be historical.
