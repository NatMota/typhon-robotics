"""Prepare finite, balanced semantic-trajectory sampling; no training or model changes."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import sys

REPRESENTATIONS = {"single": 1, "pair": 2, "triple": 3, "intent": 8}
SPLITS = {"P_train": {1, 2, 3, 5, 12, 13}, "P_validation": {6, 14, 16}}
VIEWS = {
    "overhead_main": {"slot0_array": "primary", "slot0_camera": "overhead", "slot1_array": "secondary", "slot1_camera": "oblique", "crop_source_array": "primary"},
    "oblique_main": {"slot0_array": "secondary", "slot0_camera": "oblique", "slot1_array": "primary", "slot1_camera": "overhead", "crop_source_array": "secondary"},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def group_key(row):
    return row["recording_id"], row["representation"], row["camera_variant"]


def balanced(rows, seed):
    """Visit all candidates in each take/length/view group; cycle to the largest group."""
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    quota = max(map(len, groups.values()))
    rng = random.Random(seed)
    plan = []
    for key in sorted(groups):
        selected = []
        while len(selected) < quota:
            cycle = groups[key][:]
            rng.shuffle(cycle)
            selected.extend(cycle[:quota - len(selected)])
        plan.extend(selected)
    rng.shuffle(plan)
    return [dict(draw_index=i, **row) for i, row in enumerate(plan)], quota


def summarize(rows):
    unique = {r["example_id"] for r in rows}
    return {
        "draws": len(rows), "unique_camera_examples": len(unique),
        "unique_semantic_trajectories": len({r["sample_id"] for r in rows}),
        "repeat_draws": len(rows) - len(unique),
        "per_take": dict(sorted(Counter(r["recording_id"] for r in rows).items())),
        "per_representation": dict(sorted(Counter(r["representation"] for r in rows).items())),
        "per_camera_variant": dict(sorted(Counter(r["camera_variant"] for r in rows).items())),
        "per_category": dict(sorted(Counter(r["category"] for r in rows).items())),
        "per_take_representation_view": dict(sorted(Counter("|".join(group_key(r)) for r in rows).items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/personal-m3-20261009/data-canonical/dataset.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("results/personal-m7-data-prep-20261009/temporal/source_snapshot.json.gz"))
    parser.add_argument("--semantic-index", type=Path, default=Path("results/personal-m7-data-prep-20261009/temporal/semantic-v1/index.csv"))
    parser.add_argument("--camera-audit", type=Path, default=Path("results/personal-m7-data-prep-20261009/sampling/camera_contract_verification.json"))
    parser.add_argument("--output", type=Path, default=Path("results/personal-m7-data-prep-20261009/sampling/semantic-v1"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alternative-seed", type=int, default=29)
    args = parser.parse_args()
    if args.seed == args.alternative_seed:
        parser.error("Alternative seed must differ.")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a new directory to preserve the frozen outputs.")
    manifest_hash = sha(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    episodes = {e["id"]: e for e in manifest["episodes"] if e["split"] in SPLITS}
    with gzip.open(args.snapshot, "rt") as stream:
        snapshot = json.load(stream)
    assert snapshot["dataset_sha256"] == snapshot["dataset_sha256_after_read"] == manifest_hash
    source = {e["id"]: e for e in snapshot["episodes"]}
    assert set(source) == set(episodes)
    for split, takes in SPLITS.items():
        assert {int(e["source_demo_id"].split("_")[-1]) for e in episodes.values() if e["split"] == split} == takes
    ledger = {e["id"]: e for e in manifest["eligibility"]}
    for identity, episode in episodes.items():
        e = source[identity]
        assert e["verified_source_sha256"] == e["source_sha256_after_read"] == episode["sha256"]
        assert ledger[episode["source_demo_id"]]["success"] and ledger[episode["source_demo_id"]]["split"] == episode["split"]
        assert episode["category"] in {"normal_feeding", "awkward_pause_resume_feeding"}
        assert e["frame_action_indices"] == list(range(0, episode["actions"], 4))
        assert all(len(t) == 20 for t in e["targets"])
    camera = json.loads(args.camera_audit.read_text())
    assert camera["status"] == "PASS" and camera["native_processor"] and not camera["model_loaded"]
    assert camera["source_dataset_sha256"] == manifest_hash and len(camera["probes"]) == 18
    for file in camera["source_files"].values():
        assert hashlib.sha256(file["text"].encode()).hexdigest() == file["sha256"]
    observation_file = Path("supervisor/adaptation/observations.py")
    assert sha(observation_file) == camera["source_files"][str(observation_file)]["sha256"]
    assert set(camera["camera_variants"]) == set(VIEWS)

    semantic = read_csv(args.semantic_index)
    assert len({r["sample_id"] for r in semantic}) == len(semantic)
    singles = defaultdict(list)
    for row in semantic:
        if row["representation"] == "single":
            singles[row["recording_id"]].append(row)
    for rows in singles.values():
        rows.sort(key=lambda r: int(r["semantic_start"]))
        assert len(rows) == 8
        assert all(int(a["action_end_exclusive"]) == int(b["semantic_start"]) for a, b in zip(rows, rows[1:]))
    primitive_lookup = {}
    for identity, rows in singles.items():
        for ordinal, row in enumerate(rows):
            ids = row["component_primitive_ids"].split("|")
            assert len(ids) == 1
            primitive_lookup[ids[0]] = (identity, ordinal, row)

    candidates = []
    prefix_counts = Counter()
    for row in semantic:
        identity = row["recording_id"]
        e = episodes[identity]
        assert row["split"] == e["split"]
        start, end = int(row["action_start"]), int(row["action_end_exclusive"])
        semantic_start = int(row["semantic_start"])
        frame = int(row["frame_index"])
        assert source[identity]["frame_action_indices"][frame] == start
        assert 0 <= start <= semantic_start < end <= e["actions"]
        assert semantic_start - start in range(4)
        assert int(row["context_prefix_commands"]) == semantic_start - start
        assert row["source_npz_sha256"] == e["sha256"]
        assert row["source_action_sha256"] == e["source_action_sha256"]
        anchors = list(map(int, row["anchor_action_indices"].split()))
        assert len(anchors) == int(row["true_anchor_count"])
        assert anchors[0] == start and anchors[-1] == end - 1
        assert all(a < b for a, b in zip(anchors, anchors[1:]))
        assert [i - start for i in anchors] == list(map(int, row["anchor_offsets_commands"].split()))
        assert all(abs(i / 20 - t) < 1e-10 for i, t in zip(anchors, map(float, row["anchor_timestamps_seconds"].split())))
        assert start == semantic_start // 4 * 4
        assert int(row["primitive_count"]) == REPRESENTATIONS[row["representation"]]
        ids = row["component_primitive_ids"].split("|")
        assert len(ids) == int(row["primitive_count"])
        pieces = [primitive_lookup[primitive] for primitive in ids]
        assert all(take == identity for take, _, _ in pieces)
        ordinals = [order for _, order, _ in pieces]
        assert ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))
        assert int(pieces[0][2]["semantic_start"]) == semantic_start
        assert int(pieces[-1][2]["action_end_exclusive"]) == end
        prefix_counts[semantic_start - start] += 1
        for variant, mapping in VIEWS.items():
            candidates.append(dict(row, example_id=row["sample_id"] + ":" + variant,
                category=e["category"], camera_variant=variant, **mapping,
                camera_frame_index=frame, source_npz=e["path"], source_npz_sha256=e["sha256"],
                crop_rule="O2 object_crop(selected main RGB), masked if undetected",
                target_frame="unchanged original robot world frame",
                view_geometry="original camera pixel axes; no rotations or flips"))

    train = [r for r in candidates if r["split"] == "P_train"]
    validation = [r for r in candidates if r["split"] == "P_validation"]
    assert len({group_key(r) for r in train}) == 6 * 4 * 2
    assert len({group_key(r) for r in validation}) == 3 * 4 * 2
    plan, quota = balanced(train, args.seed)
    replay, _ = balanced(train, args.seed)
    alternative, alternative_quota = balanced(train, args.alternative_seed)
    assert plan == replay and plan != alternative and quota == alternative_quota
    counts = summarize(plan)
    assert counts == summarize(alternative)
    assert set(counts["per_take_representation_view"].values()) == {quota}
    assert {r["example_id"] for r in plan} == {r["example_id"] for r in train}
    assert not {r["recording_id"] for r in train} & {r["recording_id"] for r in validation}
    validation.sort(key=lambda r: (r["recording_id"], list(REPRESENTATIONS).index(r["representation"]), int(r["semantic_start"]), r["camera_variant"]))
    val_groups = Counter(group_key(r) for r in validation)
    validation = [dict(draw_index=i, **row,
        metric_weight=1 / (3 * 4 * 2 * val_groups[group_key(row)])) for i, row in enumerate(validation)]
    assert abs(sum(row["metric_weight"] for row in validation) - 1) < 1e-12
    source_overlap = {}
    for field in ["sha256", "source_action_sha256", "source_episode_sha256"]:
        train_hashes = {e[field] for e in episodes.values() if e["split"] == "P_train"}
        val_hashes = {e[field] for e in episodes.values() if e["split"] == "P_validation"}
        source_overlap[field] = sorted(train_hashes & val_hashes)
    assert not any(source_overlap.values())

    report = {
        "schema": "m7-semantic-balanced-sampling-v1", "status": "PASS",
        "intent": "Semantic unitary trajectories, contiguous pairs/triples, and complete intent; supersedes fixed4/2/1-second proposal.",
        "command": sys.argv, "python": sys.version, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha(Path(__file__)), "source_manifest_sha256": manifest_hash,
        "semantic_index": str(args.semantic_index), "semantic_index_sha256": sha(args.semantic_index),
        "source_snapshot_sha256": sha(args.snapshot), "camera_audit": str(args.camera_audit), "camera_audit_sha256": sha(args.camera_audit),
        "config": {"seed": args.seed, "alternative_seed": args.alternative_seed, "finite_epochs_per_seed": 1,
            "quota_per_take_representation_view": quota, "representations": REPRESENTATIONS,
            "camera_variants": VIEWS, "robot_domains": 1, "target_encoding": "source legacy concatenated rot6D, unchanged",
            "train_strategy": "Equal weight per take, semantic representation class and declared camera variant; shuffled without-replacement cycles to common quota; globally shuffle complete examples.",
            "validation_strategy": "All unique semantic/view examples once; fixed take/representation/start/view order; optional weights equalize take/representation/view.",
            "category_strategy": "Equal take weights preserve4normal:2awkward mix; not forced category parity.",
            "semantic_boundaries": "Expert phase events and source commands, not arbitrary durations; component phases remain temporally contiguous within one take.",
            "prefix_rule": "Start observation/actions at the most recent existing synchronized RGB frame,0-3commands before semantic_start. Explicit context prefix retained; no invented intermediate frames.",
            "timing_rule": "Original20Hz commands and monotonic source order retained. Any sparse anchor timing comes from temporal export metadata.",
            "main_view_rule": "Slot0 is text-fused. Main-view variant is explicit; O2crop follows main source RGB. Canonical inference code remains unchanged."},
        "training": counts, "alternative_training": summarize(alternative), "validation": summarize(validation),
        "semantic_source_counts": dict(Counter(r["representation"] for r in semantic)),
        "context_prefix_commands_counts": dict(sorted(prefix_counts.items())),
        "validation_checks": {
            "seed17_and_seed29_deterministic": True, "different_seed_changes_training_order": True,
            "equal_take_representation_view_quota": True, "all_semantic_view_candidates_covered": True,
            "components_contiguous_same_take": True, "no_timestep_shuffle_or_cross_take_glue": True,
            "first_action_matches_same_frame_for_both_cameras": True, "complete_bounds_no_terminal_padding": True,
            "source_manifest_npz_and_view_contract_attested": True, "native_processor_camera_probe_count": 18,
            "targets_world_frame_and_original_camera_axes_preserved": True},
        "leakage_audit": {"take_overlap": [], "exact_source_hash_overlap": source_overlap,
            "canonical_split_rule": manifest["split_rule"],
            "session_independence": "NOT ESTABLISHED: canonical manifest explicitly says original recording spans some splits.",
            "positive_exclusions": "Recovery/correction, failed, already-completed, curated-excluded and synthetic takes remain outside positive pool.",
            "dependence": "Pairs/triples/intent share actions with singles; camera variants share source actions. Counts are augmented examples, not new independent demonstrations."},
        "scope": "Data preparation and finite example sampling only; no training, model/controller change, new viewpoint renders, or new robot domains.",
        "limits": ["Semantic phases describe generated expert robot commands, not measured human actions.",
            "Approach_and_align and transport_and_reorient are source composite phases; their internal constituents are not independently demonstrated.",
            "Dataset composition does not itself implement hierarchical conditioning, losses, branch selection or replanning.",
            "Balanced example counts do not equalize durations or per-command loss weights. No training budget is implied."],
    }
    assert sha(args.manifest) == manifest_hash
    output.mkdir(parents=True)
    shutil.copyfile(args.manifest, output / "source_dataset.frozen.json")
    write_csv(output / "semantic_camera_candidates.csv", candidates)
    write_csv(output / f"train_epoch_seed{args.seed}.csv", plan)
    write_csv(output / f"train_epoch_seed{args.alternative_seed}.csv", alternative)
    write_csv(output / "validation_frozen.csv", validation)
    report["artifact_sha256"] = {p.name: sha(p) for p in sorted(output.iterdir())}
    dump(output / "sampling_audit.json", report)
    lines = ["# Semantic trajectory sampling", "",
        "The user clarified semantic actions, not4/2/1-second windows. These are the active sampling files.",
        "", f'Training has {len(train)} unique camera examples from {counts["unique_semantic_trajectories"]} semantic trajectories across six takes.',
        f'One finite example epoch has {len(plan)} draws: {quota} per take × single/pair/triple/intent × overhead-main/oblique-main. Repeat draws: {counts["repeat_draws"]}.',
        f'Validation has {len(validation)} camera examples from {summarize(validation)["unique_semantic_trajectories"]} semantic trajectories across three takes, without resampling.',
        "", "Every example uses one take and synchronized cameras. Compositions contain adjacent whole primitive phases in source order. The0–3command prefix aligns the semantic boundary with the most recent real pre-action RGB frame.",
        "Both declared view orders use unchanged pixels/world-frame targets; the detail crop follows the selected main view. Slot0 is text-fused, so this is explicit main-view augmentation.",
        "", "Seeds17/29 change order while keeping all candidates and exact take/representation/view quotas. Validation order is independent of seed and targets.",
        "The normal/awkward mix stays4:2. No recovery, failed, already-completed, excluded or synthetic takes enter the positive pool.",
        "", "**Recording-session independence is not established:** the canonical source manifest says the same original recording spans some splits.",
        "Overlapping compositions and camera variants do not add independent demonstrations. Separate semantic exports do not implement a hierarchical policy.",
        "", "Reproduce from repository root with a new output directory:", "", "```sh",
        "python3 scripts/m7_prepare_sampling.py --seed 17 --alternative-seed 29 --output 'results/personal-m7-data-prep-20261009/sampling/semantic-replay'",
        "```", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "training": counts, "validation": summarize(validation)}, indent=2))


if __name__ == "__main__":
    main()

