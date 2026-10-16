# Published media restoration, 16 October 2026

The live page is https://www.natslabs.com/typhon-robotics/.

Restored 38 previously pending files from the verified source archive. All 38 match the SHA-256 values in the original public manifest. They are served as static files at the original media URLs; the other 381 files continue through the existing exact-key private-R2 proxy. No bucket-wide public access or new credentials were required.

The publication has 419 files, including 162 videos. The [live HTTP check](live-http-verification-20261016.json) verified status 200 and exact byte size for 419/419 media URLs. Preview checks verified 206 byte-range responses for a restored learned rollout and an existing expert replay. Live Chrome playback was checked, as were the model filter and switching between V4-M6 and V4-M4. Camera and provenance labels changed together.

- Production deployment: `dpl_6M1VnzVnBVr9U8yRgyNufsCzwujb`.
- Previous production retained by Vercel: `dpl_FmvptBUGujUP63WAadgmunPUE7cB`.
- Code/licence/media-presentation commit: `1efeb74`.
- [Recovered file checksums](media-restoration-20261016.json).
- [Published media index](media-index.json).

The site accepts `?run=<run-id>&case=<scenario-id>` on the observatory and `?clip=<relative-media-key>` in the library. The library filters by model version and reports the run, checkpoint update, training mode, checkpoint checksum, media checksum and recorded outcome.

Representative links:

- [Privileged expert pick and place](https://www.natslabs.com/typhon-robotics/media-library.html?clip=ge/personal_demo_0001.mp4).
- [Privileged expert orientation-correction reference](https://www.natslabs.com/typhon-robotics/media-library.html?clip=ge/personal_demo_0007.mp4); M1.5 labels this source scenario `orientation_correction`.
- [V4-M6 learned pickup and placement attempt](https://www.natslabs.com/typhon-robotics/?run=v4-m6-pilot-v1&case=m7-personal_demo_0006): one verified lift, zero failed grasps, position error 11.355 mm, yaw error 85.058 degrees. Orientation failed; overall manipulation success remains false. This condition used a privileged causal held cue and is labelled accordingly.

Original media bytes, execution timestamps, scenario definitions, success tolerances and experiment outcomes are unchanged. Publication availability and presentation metadata are separate from the frozen research results. The site remains a recorded snapshot, not a live robot controller.

## New recordings, 17 October 2026

Added four visually checked recordings: native feeding and loading successes from HIER02, plus a matched native loading failure and D5-assisted success from INIT12. The library reports checkpoint hashes, outcome metrics and assistance. These selected examples are separate from aggregate EVAL36 evidence. [Provenance and links](new-runs-20261017.json). The [existing-link audit](link-audit-20261017.json) passed 419/419 URLs before adding the four new files. The library now contains 166 videos and 423 media objects.
