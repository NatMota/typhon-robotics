# M0 to M1 interface v1

Use `configs/feeding_loading/m0_contract.example.json` as the interchange shape. Its poses are synthetic smoke examples; its null dimensions are unresolved measurements.

- All physical positions are metric tabletop coordinates, all yaw angles radians, and the marked face is `top`.
- Block local axes are width `+x`, depth `+y`, height `+z`; the upper edge of the blue numeral points toward local `+y`. Yaw is a full directed angle, so 180 degrees is distinct.
- Simulator coordinates have tabletop centre `x=y=0`, `+z` upward, and provisional surface height `0.800 m`.
- Declare `physical_to_sim.rotation_rad`, `translation_xy_m` and `scale` in the experiment config. The adapter applies `sim_xy = scale * R(rotation_rad) * physical_xy + translation_xy_m` and `sim_yaw = physical_yaw + rotation_rad`. Physical dimensions stay in metres, independent of this pose calibration transform.
- Null dimensions retain labelled provisional geometry. Non-null measurements update width, depth and height independently before environment construction. Constructor validation rejects unusable geometry.
- `apply_m0_contract(config, contract, transform=None)` returns a copied resolved configuration and records the source contract and transform. Explicit poses disable reset jitter. `runner --scenario` imports before constructing geometry.
- For source-pose overrides on an existing environment, `reset(seed, scenario=contract)` supports pose changes with existing geometry; changing physical dimensions requires constructing a new instance with the imported configuration.
- Source image centroids, primary clip rotations and uncalibrated M0 tracking CSVs are not metric poses. They require calibration and a declared yaw convention before import.

M1 does not edit M0 scripts, manifests or output directories. M0 completion and human review are separate from M1 acceptance. The synthetic 45-degree loading import validates this adapter path without claiming calibrated demonstration transfer.
