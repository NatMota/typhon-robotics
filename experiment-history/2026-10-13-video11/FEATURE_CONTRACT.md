# VIDEO11 feature contract

| Field | Human source reference | Runtime robot observer | Projected-asset diagnostic |
|---|---|---|---|
| Landmark | Centroid of largest accepted visible blue component near detected orange body | Centroid of accepted visible blue component in primary or rectified secondary image | Fixed complete-mask area centroid attached to the object |
| Coordinates | Original phone pixels to annotated target homography; VIDEO10 fixed rigid registration | Primary pixels to fixed virtual target homography; secondary warped to primary first | Known local object point projected to primary then fixed target homography |
| Transform order | Detect on native RGB, then transform centroid and template direction | Primary: detect then transform. Secondary: warp RGB at assumed carried plane, detect, then transform centroid | Transform one fixed 3D point and +Y direction through known camera |
| Visibility | Visible support; missing/partial values not reconstructed into full glyph | Visible support, potentially cropped/occluded by fingers | Privileged full-glyph landmark; not an occlusion-dependent centroid |
| Timestamp | Native sequential frame IDs/PTS; source decoder hashes saved | Observation before a requested control command; RGB hash/native frame not saved | Cached body_xpos/body_xquat recorded after the preceding action, before render |
| Angle | Image-template directed angle, homography-transformed direction | Template angle in current image or rectified image | Projected local +Y axis; not a template estimate |
| Height | Human metric depth unknown | Secondary assumes EEF z + object height/2 -0.005 before OPEN | Original helper uses object centre + local height/2 +0.0002 |

These are different measurement definitions. A visible centroid is not an established rigid-object landmark. A projected fixed-point diagnostic is not automatically ground truth for that visible centroid.

The implemented numeral consists of non-collision scanline boxes with centres at body-top +0.00004 m and half-thickness0.00004 m. Its top is therefore body-top +0.00008 m, not +0.0002 m. The mask's weighted x/y area centroid agrees algebraically with the scanline positions. A separate diagnostic corrects only this deterministic120µm height discrepancy; it does not change the controller or fit an offset.

Rendering evidence restriction: VIDEO10 saves H264 display frames every4control ticks and observer features every2ticks. Selected instants are0,1.525,3.05,4.875source seconds; control ticks182,304,426,572. The first and third have no exact archived video image. The others have only compressed display frames. No selected native-RGB hash or complete cached geom/camera state was saved. Object pose and joint positions alone do not attest an exact rendered scene. No geometry-based visible mask is accepted in this audit and no full sim.forward is substituted.

The fixed-pixel depth diagnostic inverts the existing warp at the recorded rectified centroid, holds that native secondary point fixed, and changes only the ray-intersection plane. This point is explicitly an inverse-warp point, not the native image's independently computed centroid. The true printed-face plane is privileged offline diagnostic information; it is never supplied to an actor.

Frozen groups: initial stillness [0,2.5), demonstrated manipulation [2.5,4.7), pre-release tail [4.7,4.9). All736common condition/timestamp rows remain in the numerical decomposition.
