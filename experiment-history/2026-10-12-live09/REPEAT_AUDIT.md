## Provenance correction and repeatability audit — 13 October

LIVE09 repeats the R10 batch completed on 12 October at approximately 09:43 UTC. The first LIVE09 report incorrectly described R10 as unrun, because it used the older prepared-input reports rather than the later completed batch. The earlier batch is at /var/tmp/typhon-ost-sweep-20261012/luna. Its results and the original preparation artifacts remain unchanged.

A post-run audit matched the primary 24 and supplementary eight slates by their full relative paths:
- All 32 choice requests and all 32 independent categorical-reason requests are semantically identical between batches.
- All 32 choices, their reported probabilities/confidences, and all 32 categorical-reason responses match.
- Across both panels, median measured choice-call latency is 1.156 s in the earlier batch and 1.094 s in LIVE09. These are observations from two batches, not a controlled speed comparison.
- LIVE09 adds zero independent states, candidate sets or robot rollouts. Do not pool repeated slates to increase the evaluation sample size. The 11/13 A-label selection pattern and the presentation-order weakness persist on this same panel.
- Matching responses do not establish the cause of repeatability, such as deterministic inference or caching.

The original LIVE09 archive and its checksum receipt remain an immutable record of the initial run/report. This correction is separately versioned with repeat_audit.json and scripts/live09_repeat_audit.py.

Reproduce the audit without API calls:
~~~sh
python3 scripts/live09_repeat_audit.py \
  --earlier /var/tmp/typhon-ost-sweep-20261012/luna \
  --repeat /var/tmp/typhon-live09-ghost-decisions-20261012 \
  --out /var/tmp/typhon-live09-repeat-audit-reproduction
~~~
Use a new output directory; the script refuses to overwrite an existing audit.
