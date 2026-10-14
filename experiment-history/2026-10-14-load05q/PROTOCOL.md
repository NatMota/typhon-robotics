**Yes. EXECUTE LOAD05-Q: a verifier-only repair followed by the minimum saved-action restart qualification. Separate output directory and a 30-minute cap.**

The XML address strings establish a verifier defect. **They do not establish that the original restoration was otherwise correct**, because its post-restore signature was not saved. Keep the stopped LOAD05 attempt classified as **failed qualification, with zero crossover results**.

## 1. Repair serialization—not restoration

Change only the signature canonicalizer and diagnostic logging. Leave state capture/restoration, controller code, physics, observation order and numerical tolerances unchanged.

For recognized XML element types, compare deterministic **XML content**, not `str(Element)` or object addresses. Preserve tags, attributes, child order, text and relevant namespaces. Standard XML libraries provide serialization/canonicalization specifically for content-based comparison. Do not strip text, discard attributes or reorder child elements to obtain equality. [Python documentation](https://docs.python.org/3/library/xml.etree.elementtree.html?utm_source=chatgpt.com)

**Do not apply a general “remove hexadecimal addresses” filter.** Unknown opaque types must be explicitly reported rather than silently ignored. Do not carry forward the earlier `meansize` exception unless independently relevant and already declared in this verifier.

Before every assertion, persist:

- Independently captured **before and after** state records, including raw XML and numerical arrays.
- Canonical signatures and their schema version.
- Field-level differences, including missing keys, shape/dtype changes and numerical differences.

Confirm that the saved “before” record does not share mutable storage with the restored environment.

## 2. Mandatory copy-only tests

Run these before creating a simulator:

| Test | Required result |
|---|---|
| XML element versus its deep copy | New content signatures equal, despite different object identities |
| Change a geometry attribute in the copy | Signature differs; diagnostic identifies the exact field |
| Change child order or meaningful text | Difference retained—not normalized away |
| Change one element of a synthetic numerical state array | Full-state verifier detects it and saves both values |
| Unsupported opaque object | Explicit unsupported-field result, not assumed equivalence |

These tests qualify the comparator. They do **not** qualify an actual simulator restart.

## 3. At most two saved-action restart checks

Freeze the repaired verifier before these checks. Use the existing **C and B demo0012/loading/visible-fixture/seed912171** archives, branching at the already-selected **tick100/query2**.

For each source, once:

1. Replay its archived prefix to tick100.
2. Capture the complete pre-restore state.
3. Apply the **unchanged** copy/restore procedure.
4. Save the complete post-restore state and field-level comparison before asserting.
5. If the state check passes, execute **the next 30 archived actions** with the original observation/render call order; compare against the corresponding archived trace.

**No model loading or inference is needed.** No crossover, extra `forward()`, new seed or controller adjustment belongs in this qualification.

Retain full physical/controller checks, including controls, applied forces, warm-start state, cached quantities and command history. XML-content equality cannot replace those checks: MuJoCo’s continuation can depend on actuator, callback and solver state beyond positions and velocities. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

## 4. Decision gate

| Result | Action |
|---|---|
| Both content/state comparisons and saved-action continuations pass | **Restart qualification passes for the repaired verifier and these states.** |
| XML content matches but another state field differs | Save the exact difference and stop; no quiet exclusion or tolerance change |
| State signature passes but the continuation diverges | Restoration remains unqualified; preserve the first divergence |
| Tests or qualification cannot finish within 30 minutes | Close with the specific blocker; no repair-and-retry loop |

**A successful new qualification does not retroactively prove that pointer strings were the original attempt’s only discrepancy.**

On a pass, **proceed without another approval round to the already-authorized LOAD05 pre-contact crossover**, in a new, separately frozen execution namespace: the same two states, C/B checkpoints, paired noise and maximum four 90-step continuations. Preserve its original scientific question and existing execution cap; add no conditions.

The immediate work is therefore **content-safe signatures → recorded before/after evidence → two restart checks**. This repairs a demonstrated measurement defect without weakening the physical gate or turning a failed restart into model evidence.