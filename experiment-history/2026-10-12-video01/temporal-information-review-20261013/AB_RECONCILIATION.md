# VIDEO01-A/B — Administrative evidence reconciliation
13 October 2026. This closes two evidence reviews, not the unmet robot-control capabilities.

## VIDEO01-A (#11016)
Accept: object-image representation implemented and audited, with explicit limits.
- The original sparse M0 lineage has526 sample IDs,440 accepted centres and227 accepted primary yaw estimates. Its source-index/frame hashes remain attached. Coverage is not pose accuracy.
- The separately regenerated dense-v2 lineage has2644 sequentially verified native frame identities. Superseded dense-v1's906 mismatches stay preserved; dense-v2 verification is not silently attributed to a different old export.
- Values are image observations of visible support/centroids, not a fixed physical block centre under occlusion. Lost or low-confidence observations and long gaps remain unknown; path fragments can have a shorter summed length than total endpoint displacement because missing intervals are not joined.
- VIDEO06/07 show useful native glyph-feature motion while retaining failed support. This is image motion, not metric XYZ.
- VIDEO08 establishes that the available target annotations admit substantially different camera hypotheses; tabletop homography is not a validated3D lift reconstruction.
- VIDEO09–11 distinguish actor RGB observations from the independent projected complete-glyph landmark scorer. The latter does not retroactively certify the visible observer or a full-path advantage.
- Remaining capability: calibrated/qualified human-to-robot action mapping remains blocked under#10989 and calibration under#10991.

## VIDEO01-B (#11017)
Accept: hand representation reviewed; not qualified for autonomous grasp/release.
-526 sparse observations retain175 hand detections. Hand presence is not recall; many frames deliberately have no hand. Handedness confidence is not landmark accuracy.
- Model-world landmarks are hand-centred predictions, not calibrated table/robot coordinates. Pinch aperture and overlap do not establish forces, support transfer, successful grasp or metric lift.
- VIDEO03–05 add limited held-state probe evidence on disclosed labels. VIDEO05 repairs some challenge errors but retains an excluded-take approach false positive; withdrawal/approach cases and uncertain views remain in the record.
- Continued finger support on selected frames is not independently qualified stationary-held robustness. VIDEO07 phase brackets inherit sampling and sync uncertainty.
- All original sparse hand outputs retain unknown is_grasped and global metric pose. This means no qualified output was supplied; it does not prove the images or embeddings contain no grasp information.
- Remaining capability: observation authority and reliable causal contact/release remain blocked under#10984; action alignment remains under#10989.

## Supporting evidence
- [Initial representation report](https://github.com/NatMota/typhon-robotics/blob/research/video01-visual-features/experiment-history/2026-10-12-video01/REPORT.md)
- [Temporal-information acceptance review](REPORT.md)
- [VIDEO05 hard-negative and held-pause audit](https://github.com/NatMota/typhon-robotics/blob/research/video05-precontact-pause/experiment-history/2026-10-12-video05/REPORT.md)
- [VIDEO07 native-cadence evidence](https://github.com/NatMota/typhon-robotics/blob/research/video07-native-cadence/experiment-history/2026-10-12-video07/REPORT.md)
- [VIDEO08 calibration limits](https://github.com/NatMota/typhon-robotics/blob/research/video08-calibration-identifiability/experiment-history/2026-10-12-video08/REPORT.md)
- [VIDEO11 feature contract and diagnostic limitations](https://github.com/NatMota/typhon-robotics/blob/research/video11-feature-correspondence/experiment-history/2026-10-13-video11/REPORT.md)

This reconciliation ran no new extraction, classifier fitting, physics or robot control. The small AB_RECONCILIATION_CHECKS.json checks original row identity/counts/unknown fields and records supporting-file hashes. It is not a new visual accuracy test.
