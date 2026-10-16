# RESTORE-GOAL: verified checkpoint restoration

The exact GOAL-varied checkpoint was restored from the preserved Mint archive to Powerglide, with a matching full destination SHA-256 and file size. Restoration took 46.91 minutes within its separate 90-minute allowance. This was infrastructure work: zero policy evaluations, simulation episodes or training updates.

| Check | Result |
|---|---|
| Bytes | 3,729,340,908 |
| SHA-256 | d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb |
| Completed UTC | 2026-10-16T05:20:56.303403+00:00 |
| Finalized path | /var/tmp/typhon-ost-sweep-20261012/goal-grounding/train-varied/step3000.pt |
| Free bytes after restoration | 1,276,448,768 |
| Atomic final rename after checksum | Yes |
| Original HIER02 outcomes | Zero; preparation failure preserved |

The 638,189,568-byte partial from the original preparation attempt was moved outside loader paths, resumed with rsync --append-verify, then fully read and hashed at the destination. No model substitution or paid machine was used. The source archive remained unchanged.

HIER02-LIVE-R1 is a separate activation using the original scientific runner and fixed eight workflows. Its clock starts after this verified restore. See ticket11093 and the private branch research/hier02-live-r1-20261016. The original ticket11090 remains preparation-blocked, and the reference-state audit is ticket11092.

Files in this directory preserve the source manifest, original restore clock, command, transfer log, progress snapshots and final verification receipt.
