# DUPLO 31110pb074 visual v2

The orange brick lies on its side with its printed face pointing along local +Z. The top of the blue 2 and the brick's studs point along local +Y. This is the frame used by M0's user-confirmed upright convention and M1.5's reconstruction. Rotating the block rotates all these features together.

`numeral_2_mask.png` is copied byte-for-byte from the personally recorded M0 blue-numeral template. See `provenance.json` for the frame and checksum. It retains the curved, non-centrosymmetric shape of the real print. It is rendered as flat opaque scanline geometry, not an external font or a seven-segment approximation. There is no block texture/UV map to mirror or wrap onto the wrong face.

Four visual-only studs mark the physically present stud end; their proportions are approximate. The visual body reserves the stud projection inside the original envelope. The underside cavity and exact surface finish are not modelled. Collision remains the old full rectangular envelope, with identical mass, inertia and friction. Every added visual geom has zero mass and zero collision masks.

Select `object.visual_version = "duplo_31110pb074_v2"` explicitly. Missing version fields preserve the legacy appearance. New example configs use the user-confirmed part's catalogue envelope (32×32×43 mm, lying printed-side-up as 32×43×32 mm) and 12.7 g mass, retained from M1.5. These are catalogue estimates, not measurements of the individual piece: https://www.bricklink.com/v2/catalog/catalogitem.page?P=31110pb074

At native 512×512 camera resolution, all four orientations pass the saved pixel tests. Do not assume shrinking the entire workspace to 336/256/224/128 pixels preserves numeral legibility. The report records each resolution; a future policy-specific crop or preprocessing path needs its own validation.
