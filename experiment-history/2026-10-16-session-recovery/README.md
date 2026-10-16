# Recovery package for the 16 October session

The existing [research report](../2026-10-12-ost-sweep/REPORT.md) contains the short session update. The existing [claim index](../2026-10-13-del02-package/CLAIM_INDEX.md) now has seven additional entries, C15–C21. Earlier text, claims and scores remain preserved.

[INIT11](INIT11_PROTOCOL.md) is a prepared future protocol with **zero executions**. It does not authorize another experiment tonight. Its exact query-4 handback points and masked previous-target input are in [handback-boundaries.json](handback-boundaries.json). Native noise resumption and a verified future compute window remain activation prerequisites.

[recovery_receipt.json](recovery_receipt.json) identifies immutable source commits, verified Mint archives, checkpoint locations, media limitations and unexecuted work. Verification reuses prior successful large-file receipts; no multi-gigabyte model was rehashed just for this package. The older full workstation backup was a live filesystem copy, not an atomic OS image or a fresh restore test.

The new review verifies 32 source references, 18 numerical cells, both handback boundaries and the package’s relative links. Reproduce the read-only checks from the repository root:

~~~sh
python3 scripts/verify_session_recovery_20261016.py \
  --repo . --output /tmp/typhon-session-recovery-check.json
~~~

The source commits must be available in the Git object store and the listed Mint archives mounted. A missing source is an explicit prerequisite, not permission to silently use a different version. The executed boundary audit source is retained for inspection; its fixed historical output path is not a new experiment launcher.

The programme remains active. The current compute session is preserved before the conservative 01:00 UTC / 02:00 BST cutoff. No extension, public publication or submission is claimed.
