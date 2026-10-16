# EVIDENCE24 — One verified checkpoint copy removed from HOME

Completed work, ticket #11099. Nat's exact-path authorization permitted removing one inactive research checkpoint after fresh source/archive hashes and identity, mount and process checks. No research jobs were started.

Powerglide's persistent HOME had reached its 100 GiB hard quota. Local scratch capacity from EVIDENCE23 could not resolve this separate quota.

| Measure | Observed result |
|---|---:|
| HOME quota use immediately before | 104,857,472 KiB |
| HOME quota use immediately after | 101,412,896 KiB |
| Hard quota | 104,857,600 KiB |
| Exact quota bytes recovered | 3,527,245,824 |
| Logical source file bytes | 3,527,238,808 |
| Source filesystem allocated bytes | 3,527,245,824 |
| Headroom immediately after | 3,527,376,896 bytes |

Quota changed slightly during the preceding inventory as other activity continued. The table uses the immediate before/after measurements around this single unlink; it does not substitute the earlier rounded quota display.

## Exact scope and verification

Removed source:

`/cs/student/msc_rai/2026/ndossant/Projects/humanoid-m6-joint/results/personal-m7-data-prep-20261009/training-integration-v2/m3-dense-probe-01/m7-step000001.pt`

Retained archive:

`/media/nm/Fast Drive/Typhon/archives/powerglide-full-20261014/home/Projects/humanoid-m6-joint/results/personal-m7-data-prep-20261009/training-integration-v2/m3-dense-probe-01/m7-step000001.pt`

Both fresh complete SHA-256 reads matched `8004ac5f60a9107c9d521368b6d17aa52eb2ef209be4cd44a5503dd5268c0835`. No duplicate hash job was launched after authorization; the already-running source and destination checks were used. Each hash operation confirmed stable size, inode and modification time. Before deletion the pinned device/inode, size, modification/change times, ownership, mode and link count were checked again, including an immediate directory-file-descriptor identity check.

| Identity field | Source immediately before | Mint archive after |
|---|---:|---:|
| Device | 42 | 2114 |
| Inode | 2708838567 | 6746429 |
| Bytes | 3527238808 | 3527238808 |
| Modification time, ns | 1791575578322975651 | 1791575578322975600 |

Nanosecond values in copied receipts remain the original Python JSON integers; they were copied as bytes and must not pass through a lossy JavaScript number conversion. The 51 ns source/archive timestamp difference reflects archive filesystem precision; each copy was guarded against its own unchanged metadata, and content equality was established with SHA-256.

The archive was open read-only and pinned during removal. Fast Drive was confirmed mounted from `/dev/sde2` at `/media/nm/Fast Drive`, owned by `nm:nm`. Its metadata was unchanged afterward. The archive manifest is `/media/nm/Fast Drive/Typhon/archives/powerglide-full-20261014/SHA256SUMS-reconciled.jsonl`, with reconciliation receipt `/media/nm/Fast Drive/Typhon/archives/powerglide-full-20261014/backup-reconciled-verification.json`.

No file-descriptor, memory-map or command-line references were observed among 93 own-user processes. Protected authentication processes could not be fully inspected; their identities are recorded. The checkpoint is a historical one-update smoke artifact, not an active policy or current lab dependency. Historical M4 reproduction still refers to it and requires restoration from the retained archive when replayed.

## Preserved material and failed preflights

The M4 checkpoint, all eight historical M3 sidecars, environments, active processes, Mint archive and manifests were preserved. Only the exact source path was unlinked.

Before the successful operation, guards/transport failed without deletion: a JavaScript nanosecond-rounding mismatch, a remote-login-shell parsing error, and a transient quota-command failure. The source was checked unchanged; integer transport was made lossless, Python was sent via the established stdin route, and quota was queried specifically for HOME. Identity checks were not relaxed. Saved preflight/failure receipts accompany the successful receipt.

The first timestamp mismatch occurred before a receipt file was created; its precise diagnosis is retained in the conversation tool record. The subsequent two failed attempts and successful attempt have durable receipts here.

## Receipt and limits

Authoritative receipt: `/media/nm/Fast Drive/Typhon/storage-cleanup-20261016/m3-single-file-092728-deletion-receipt.json`.

Receipt SHA-256: `45ce7d63bf1ca88ac00a3814cc0025a1065187f8665013872e48d399cc4f0a0b`.

This records quota recovery, not a lab smoke-test result. The supervisor was to independently verify lab writes. No current or frozen experiment result was recomputed.
