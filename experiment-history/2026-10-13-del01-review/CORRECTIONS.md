# DEL01: private media and claim review

The eight deterministically selected evidence records are traceable to their original episode/configuration, model or assisted controller, action record and video. All eight originals and both existing VIDEO10 comparison copies decode fully and have identical hashes on Powerglide and Mint. Their archived evidence is qualified; **none has an exact matching entry in the audited hosted index**.

This review updates private captions, the README headline and a dashboard context note. It does not deploy the site, upload videos, change access permissions, regenerate media or run a robot.

## What the clips actually show

First eligible frozen scenario ID was used for each slot. The fixed control uses the same configuration and seed as the selected varied success. All eight start unfinished.

| Record | Final centre error | Directed yaw error | Endpoint and meaning |
|---|---:|---:|---|
| Broad near-layout success | 6.88 mm | 1.28° | Original strict success; panel 17/40 versus vanilla 0/40 |
| Varied new-combination success | 4.00 mm | 1.31° | Original strict success; panel 10/12 |
| Matched fixed continuation | 180.52 mm | 0.78° | Outside target despite correct yaw; panel 0/12 |
| Varied loading failure | 263.27 mm | 1.65° | Wrong destination, marked face not up, not seated; panel 0/12 |
| Original V3 film-demo0002 | 19.61 mm | 2.70° | Released/stable, upright and contained; misses original 15 mm centring only |
| Original V4 film-demo0006 | 11.42 mm | 85.33° | Pickup/placement partial result; fails orientation |
| VIDEO10 SEQ | 10.89 mm | 2.11° | Successful assisted correction |
| VIDEO10 STATIC | 10.94 mm | 0.40° | Same task succeeds without intermediate sequence reference |

All eight selected records have a saved positive pickup/lift indicator. The original V3/V4 records use the legacy 70 mm pickup definition; the newer policy records use a 10 mm lift with simultaneous bilateral contact for three consecutive 20 Hz steps. These definitions are retained separately in the manifest.

Low yaw error is not a complete pose or task score. The manifest retains every original criterion, including upright face, footprint, release, stability and loading seating.

The main controlled positive result is varied versus equal-update fixed training. V4 remains a useful grasp-relative failure case. V3's oriented in-square placement is a labelled post-hoc region result; it does not change the original endpoint or statistical tests. Broad near-layout pooled 42.5% is kept separate from the family-balanced 41.67% estimate and its 22.22–55.56% interval.

## Identity, assistance and timing

The four newer policy clips use the exact broad/fixed/varied SHA identities from EVAL02R. The two original clips use the pinned 775 suite hashes and episode provenance. They are not substituted 9 October pilot videos. Original V3/V4 have privileged causal held-state conditioning; their displayed camera replay is not a dump of every policy input.

VIDEO10 uses an RGB visual-servo adapter, not X-VLA. Both records include the same 142-command supplied acquisition/alignment prefix, registered goal reference and reviewed release schedule. The seven-point strict scoring remains independent of the actor. Human footage is held at its starting frame during the prefix and then shown at 4× slower source time. The derived videos contain condition/time text and two phone views above two robot views. Their saved 193-entry frame clocks, source provenance and derivation code are linked; the private captions make that assistance explicit.

The six policy clips are 73.4 seconds at 5 fps, 367 decoded frames each. The two assisted originals and two comparison copies are 38.6 seconds at 5 fps, 193 frames each. Simulation time differs slightly from encoded duration because of capture boundaries; inference wall time is separate. No throughput or reaction-time inference is made from playback speed.

An agent inspected decoded start/end and representative intermediate frames, including both VIDEO10 comparison views. This is not a claim of human review or manual inspection of every frame. Full-file decoding and independent pose-error arithmetic supplement the saved independent task scorer. The fixed/varied pair has identical initial physical pose, scenario and seed.

## Corrected private claims

- Replaced the old global pilot TL;DR with bounded broad-Pt and varied-goal results.
- Separated the historical 9 October pilot table from current evidence.
- Kept original 775 strict differences as completed-state preservation and preserved V3's genuine partial progress.
- Removed the implication that original V4's correction-data benefit is established; its extra 500 updates and missing three recovery arrays remain explicit.
- Stated that the broad-six episodes are a catalogue subset, not six new recordings, and that source/development takes share recording pairs.
- Kept numeric-language grounding unproven because image/goal information co-varies.
- Distinguished assisted full-sequence execution from an incremental benefit over STATIC/END.
- Kept 68 intervention trials as unexecuted.
- Qualified the historical completion guard and service description against later observer/control failures.

The scoped inspection covered README, dashboard HTML/documentation and the currently hosted dashboard, state payload, script, media library and index. It was not a rewrite of frozen historical reports.

## Delivery gaps

The current site returned HTTP 200 and accurately labels itself a 9 October 22:53 UTC snapshot. Its index contains 419 media entries, with 381 published and 38 pending according to that index. These are indexed counts, not a new availability test of all 419 objects.

None of the eight selected originals, or the two associated comparison copies, matches a hash in that index. Their current hosted accessibility therefore remains unverified; no public URL is invented. Two related older V4 pilot URLs still return 404. They have different hashes/executions and are excluded as substitutes.

The existing dashboard importer expects summary objects with results[]; later evaluations use a different row-list layout. This review does not silently convert those schemas or imply that the hosted dashboard now includes these clips. The new private manifest/captions are the qualified handoff for a future explicitly scoped delivery update.

Root PUBLIC_SHA256SUMS and EXPORT_PROVENANCE.json describe the original export and were already stale for the current README. They remain unchanged. This review retains before/after document hashes and its own receipts rather than rewriting historical provenance.

Other scientific gaps remain: no recording-session holdout, one fit seed, original V4 correction-array availability, the narrow initializer-header limitation, unexecuted interventions and no established incremental full-sequence motor-transfer benefit.

## Reproduce and locate

```sh
python3 scripts/del01_media_review.py --root "/media/nm/Fast Drive/Typhon" --output "/media/nm/Fast Drive/Typhon/research-20261012/del01-media-claims-review/reproduced"
```

The command reads preserved inputs and cached, timestamped access evidence; it does not launch inference, physics or hosting. A fresh availability check would need a new timestamped receipt. The exact read-only Powerglide full-decode/hash script is saved in inputs/powerglide_media_probe.py; it uses the saved selected_records.json as its input.

- Canonical media/claim manifest: [media_claim_manifest.json](media_claim_manifest.json).
- Dashboard copy: [private captions](../../dashboard/review/media_claim_manifest.json).
- Input hashes: [checksum_receipt.json](checksum_receipt.json).
- Before/after private text: [claim_changes.json](claim_changes.json).
- Current access evidence, deterministic selection and Powerglide full-decode results: [inputs](inputs/).
- Broader comparison interpretation: [EVAL02R](../2026-10-13-eval02-review/CLAIM_STATUS.md).

DEL01's bounded evidence/claim review is complete with explicit hosting gaps. Publication, media upload, importer integration and a polished submission package are separate from this result.
