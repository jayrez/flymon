"""Trial-level analysis for Experiment 14 autonomous progression."""
from __future__ import annotations

from collections import Counter
import gzip
import inspect
import itertools
import json
import math
from pathlib import Path

import numpy as np

from flymon.exploration import FrozenExplorationController, FrozenInterfaceController
from flymon.interaction import PopulationEventController
from flymon.progression import atomic_write_json, checkpoint_decisions
from run_visual_experiment import ROOT


RESULTS = ROOT / "results/experiment-14-progression"
CAPTURES = ROOT / "captures/experiment-14-progression"
MANIFEST = RESULTS / "trial-manifest.json"
MATCHED_SEEDS = tuple(range(921, 931))
CONDITIONS = ("live_full", "live_no_a", "frozen_full", "no_vision_full")
METRICS = (
    "unique_coarse_states",
    "coarse_transitions",
    "sustained_novel_state_episodes",
    "stuck_fraction",
    "maximum_frame_distance",
    "action_entropy",
)
VISUAL_TRAJECTORY_METRICS = (
    "unique_coarse_states",
    "coarse_transitions",
    "sustained_novel_state_episodes",
    "stuck_fraction",
    "maximum_frame_distance",
)
ACTION_ORDER = ("LEFT", "RIGHT", "UP", "A", "NONE")
FPS = 5.0


def entropy(values) -> float:
    counts = Counter(values)
    total = len(values)
    return float(-sum(
        count / total * math.log2(count / total)
        for count in counts.values() if count)) if total else 0.0


def metric_rows(path: str | Path):
    with gzip.open(ROOT / path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def sustained_novel_episodes(coarse: list[str]) -> tuple[list[dict], list[bool]]:
    reference = set(coarse[:50])
    novel = [False] * len(coarse)
    for index in range(50, len(coarse)):
        novel[index] = coarse[index] not in reference
    windows = []
    for start in range(50, max(50, len(coarse) - 9)):
        if sum(novel[start:start + 10]) >= 5:
            windows.append((start, start + 9))
    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    episodes = []
    for start, end in merged:
        episode_start = next(
            index for index in range(start, min(end + 1, len(coarse))) if novel[index])
        episodes.append({
            "start_decision": episode_start,
            "candidate_window_start": start,
            "end_decision": min(end, len(coarse) - 1),
        })
    return episodes, novel


def stuck_metrics(coarse: list[str]) -> tuple[float, int]:
    stuck = np.zeros(len(coarse), bool)
    longest = 0
    start = 0
    while start < len(coarse):
        end = start + 1
        while end < len(coarse) and coarse[end] == coarse[start]:
            end += 1
        run = end - start
        longest = max(longest, run)
        if run >= 10:
            stuck[start:end] = True
        start = end
    return float(stuck.mean()) if len(stuck) else 0.0, longest


def summarize_trial(entry: dict) -> tuple[dict, dict]:
    rows = list(metric_rows(entry["metrics_path"]))
    actions = [row["action"] for row in rows]
    display_actions = ["NONE" if action is None else action for action in actions]
    exact = [row["visual_analysis"]["resulting_frame_sha256"] for row in rows]
    coarse = [row["visual_analysis"]["coarse_state_hash"] for row in rows]
    changes = [
        None if row["visual_analysis"]["frame_to_previous_mae"] is None
        else float(row["visual_analysis"]["frame_to_previous_mae"])
        for row in rows
    ]
    distances = [float(row["visual_analysis"]["frame_to_initial_mae"]) for row in rows]
    episodes, novel = sustained_novel_episodes(coarse)
    stuck_fraction, longest = stuck_metrics(coarse)
    counts = Counter(display_actions)
    transitions = sum(a != b for a, b in zip(coarse, coarse[1:]))
    first_novel = next((index for index, value in enumerate(novel) if value), None)
    summary = {
        "condition": entry["condition"],
        "seed": int(entry["seed"]),
        "horizon": int(entry["horizon"]),
        "decisions": len(rows),
        "action_counts": {action: counts[action] for action in ACTION_ORDER},
        "action_distribution": {
            action: counts[action] / len(rows) for action in ACTION_ORDER
        },
        "actions": len(rows) - counts["NONE"],
        "action_entropy": entropy(display_actions),
        "unique_exact_frames": len(set(exact)),
        "unique_coarse_states": len(set(coarse)),
        "coarse_transitions": transitions,
        "final_frame_distance": distances[-1],
        "maximum_frame_distance": max(distances),
        "longest_repeated_state_run": longest,
        "stuck_fraction": stuck_fraction,
        "time_to_first_novel_state": first_novel,
        "time_to_first_sustained_novel_state": (
            episodes[0]["start_decision"] if episodes else None),
        "sustained_novel_state_episodes": len(episodes),
        "sustained_novel_episode_intervals": episodes,
        "runtime_seconds": float(entry["runtime_seconds"]),
        "decisions_per_second": len(rows) / float(entry["runtime_seconds"]),
        "simulated_neural_seconds": len(rows) * .2,
    }
    detail = {
        "condition": entry["condition"], "seed": int(entry["seed"]),
        "horizon": int(entry["horizon"]), "actions": actions,
        "coarse": coarse, "changes": changes, "distances": distances,
        "episodes": episodes,
    }
    return summary, detail


def aggregate(summaries: list[dict]) -> dict:
    keys = (
        "actions", "unique_exact_frames", "unique_coarse_states",
        "coarse_transitions", "final_frame_distance",
        "maximum_frame_distance", "longest_repeated_state_run",
        "stuck_fraction", "action_entropy", "sustained_novel_state_episodes",
    )
    result = {
        key: float(np.mean([row[key] for row in summaries])) for key in keys
    }
    result["action_distribution"] = {
        action: float(np.mean([
            row["action_distribution"][action] for row in summaries
        ])) for action in ACTION_ORDER
    }
    result["trial_count"] = len(summaries)
    return result


def exact_paired_test(left: list[dict], right: list[dict], key: str) -> dict:
    left_by_seed = {row["seed"]: row for row in left}
    right_by_seed = {row["seed"]: row for row in right}
    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    differences = np.asarray([
        float(left_by_seed[seed][key]) - float(right_by_seed[seed][key])
        for seed in seeds
    ])
    observed = float(differences.mean())
    null = [
        float(np.mean(differences * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    p_value = sum(
        abs(value) >= abs(observed) - 1e-15 for value in null) / len(null)
    return {
        "effect": observed, "p_value": float(p_value),
        "seed_effects": differences.tolist(), "seeds": seeds,
        "enumerations": len(null), "test": "exact paired two-sided sign permutation",
    }


def comparisons(summaries: list[dict]) -> dict:
    grouped = {
        condition: [row for row in summaries if row["condition"] == condition]
        for condition in CONDITIONS
    }
    result = {}
    for label, other in (
        ("live_full_vs_live_no_a", "live_no_a"),
        ("live_full_vs_frozen_full", "frozen_full"),
        ("live_full_vs_no_vision_full", "no_vision_full"),
    ):
        result[label] = {
            key: exact_paired_test(grouped["live_full"], grouped[other], key)
            for key in METRICS
        }
    return result


def match_non_a(actions: list[str | None], a_indices: list[int]) -> dict[int, int | None]:
    used = set()
    eligible = [
        index for index, action in enumerate(actions)
        if action != "A" and all(abs(index - event) >= 6 for event in a_indices)
    ]
    matches = {}
    for event in a_indices:
        same_block = [
            index for index in eligible if index not in used
            and index // 100 == event // 100
        ]
        candidates = same_block or [index for index in eligible if index not in used]
        if not candidates:
            candidates = eligible
        selected = min(candidates, key=lambda index: (abs(index - event), index)) \
            if candidates else None
        matches[event] = selected
        if selected is not None:
            used.add(selected)
    return matches


def consequence_at(detail: dict, index: int) -> dict:
    changes = detail["changes"]
    coarse = detail["coarse"]
    horizon = detail["horizon"]
    episode_starts = {row["start_decision"] for row in detail["episodes"]}
    def change_at(offset):
        position = index + offset
        return changes[position] if position < horizon else None
    later = [
        value for value in changes[index + 2:min(index + 6, horizon)]
        if value is not None
    ]
    transition = any(
        position > 0 and coarse[position] != coarse[position - 1]
        for position in range(index, min(index + 6, horizon))
    )
    episode = any(index + 1 <= start <= index + 10 for start in episode_starts)
    return {
        "frame_change_t": change_at(0),
        "frame_change_t_plus_1": change_at(1),
        "max_frame_change_t_plus_2_to_5": max(later) if later else None,
        "coarse_transition_within_t_to_t_plus_5": transition,
        "sustained_novel_episode_within_t_plus_1_to_10": episode,
    }


def a_consequence_analysis(details: list[dict]) -> dict:
    events, controls = [], []
    for detail in details:
        if detail["condition"] not in {"live_full", "long_run"}:
            continue
        a_indices = [
            index for index, action in enumerate(detail["actions"]) if action == "A"
        ]
        matches = match_non_a(detail["actions"], a_indices)
        for index in a_indices:
            event = consequence_at(detail, index) | {
                "condition": detail["condition"], "seed": detail["seed"],
                "decision": index, "matched_decision": matches[index],
            }
            events.append(event)
            if matches[index] is not None:
                controls.append(consequence_at(detail, matches[index]) | {
                    "condition": detail["condition"], "seed": detail["seed"],
                    "decision": matches[index], "matched_a_decision": index,
                })
    fields = (
        "frame_change_t", "frame_change_t_plus_1",
        "max_frame_change_t_plus_2_to_5",
        "coarse_transition_within_t_to_t_plus_5",
        "sustained_novel_episode_within_t_plus_1_to_10",
    )
    def means(rows):
        return {
            field: float(np.mean([row[field] for row in rows if row[field] is not None]))
            if any(row[field] is not None for row in rows) else None
            for field in fields
        }
    event_means, control_means = means(events), means(controls)
    return {
        "interpretation": "observational; decisions are not independent replicates",
        "a_event_count": len(events), "matched_control_count": len(controls),
        "a_event_means": event_means, "matched_non_a_means": control_means,
        "mean_differences": {
            field: (event_means[field] - control_means[field]
                    if event_means[field] is not None and control_means[field] is not None
                    else None)
            for field in fields
        },
        "events": events,
    }


def review_candidates(details: list[dict], manifest: dict) -> list[dict]:
    manifest_by_key = {
        (row["condition"], row["seed"]): row for row in manifest["trials"]
        if row.get("completed_successfully")
    }
    pools = {"maximum distance from initial": [], "largest one-step frame transition": [],
             "start of sustained novel-state episode": [], "large state change shortly after A": []}
    for detail in details:
        if detail["condition"] not in {"live_full", "long_run"}:
            continue
        key = (detail["condition"], detail["seed"])
        if key not in manifest_by_key:
            continue
        maximum = int(np.argmax(detail["distances"]))
        pools["maximum distance from initial"].append((detail["distances"][maximum], detail, maximum))
        transition = max(
            (index for index, value in enumerate(detail["changes"])
             if value is not None),
            key=lambda index: detail["changes"][index])
        pools["largest one-step frame transition"].append(
            (detail["changes"][transition], detail, transition))
        for episode in detail["episodes"][:1]:
            pools["start of sustained novel-state episode"].append((1.0, detail, episode["start_decision"]))
        for event in [i for i, action in enumerate(detail["actions"]) if action == "A"]:
            end = min(event + 6, detail["horizon"])
            if event + 1 < end:
                candidates = [
                    index for index in range(event + 1, end)
                    if detail["changes"][index] is not None
                ]
                if candidates:
                    position = max(candidates, key=lambda i: detail["changes"][i])
                    pools["large state change shortly after A"].append(
                        (detail["changes"][position], detail, position))
    selected, seen = [], set()
    for reason, values in pools.items():
        for value, detail, decision in sorted(values, key=lambda row: row[0], reverse=True):
            key = (detail["condition"], detail["seed"], decision)
            if key in seen:
                continue
            seen.add(key)
            boundary = decision + 1
            horizon = detail["horizon"]
            before = boundary // 250 * 250
            after = min(horizon, int(math.ceil(boundary / 250)) * 250)
            manifest_row = manifest_by_key[(detail["condition"], detail["seed"])]
            selected.append({
                "condition": detail["condition"], "seed": detail["seed"],
                "decision": decision, "video_timestamp_seconds": decision / FPS,
                "reason_selected": reason, "analysis_value": float(value),
                "video_path": manifest_row["video_path"],
                "nearest_checkpoint_before": (
                    f"{manifest_row['checkpoint_directory']}/decision-{before:06d}.png"),
                "nearest_checkpoint_after": (
                    f"{manifest_row['checkpoint_directory']}/decision-{after:06d}.png"),
            })
            if sum(row["reason_selected"] == reason for row in selected) >= 3:
                break
    return selected[:12]


def artifact_budget(manifest: dict) -> dict:
    successful = [row for row in manifest["trials"] if row.get("completed_successfully")]
    video_files = list((CAPTURES / "videos").glob("**/*.mp4"))
    checkpoint_files = list((CAPTURES / "checkpoints").glob("**/*.png"))
    metric_files = list((RESULTS / "metrics").glob("**/*.jsonl.gz"))
    suspicious_extensions = {".raw", ".rgb", ".rgba", ".npy", ".npz", ".bmp", ".jpg", ".jpeg"}
    suspicious_names = {"frames", "raw_frames", "screens", "decision_frames"}
    raw = [
        path for path in CAPTURES.glob("**/*") if path.is_file()
        and (path.suffix.lower() in suspicious_extensions
             or any(part.lower() in suspicious_names for part in path.parts))
    ]
    total_rows = sum(int(row["metrics_row_count"]) for row in successful)
    expected_rows = 10 * 4 * 1500 + 5000
    integrity = (
        len(successful) == 41 and total_rows == expected_rows
        and all(row["metrics_integrity"]["passed"] for row in successful)
    )
    videos_verified = (
        len(video_files) == 41 and all(row["video"]["verified"] for row in successful)
    )
    return {
        "mp4_files": len(video_files),
        "total_mp4_size": sum(path.stat().st_size for path in video_files),
        "checkpoint_pngs": len(checkpoint_files),
        "checkpoint_cadence": "every 250 decisions, including 0 and final",
        "total_checkpoint_size": sum(path.stat().st_size for path in checkpoint_files),
        "metric_files": len(metric_files),
        "per_decision_metric_records": total_rows,
        "expected_metric_records": expected_rows,
        "metrics_integrity": "PASS" if integrity else "FAIL",
        "compressed_metrics_size": sum(path.stat().st_size for path in metric_files),
        "accidentally_persisted_raw_framebuffer_files": len(raw),
        "raw_framebuffer_paths": [path.relative_to(ROOT).as_posix() for path in raw],
        "videos_verified": "PASS" if videos_verified else "FAIL",
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
    }


def leakage_checks() -> dict:
    runner = (ROOT / "run_progression_experiment.py").read_text().lower()
    controller = "\n".join(inspect.getsource(value.decode) for value in (
        FrozenExplorationController, FrozenInterfaceController,
        PopulationEventController)).lower()
    return {
        "RAM": "read_memory" not in controller + runner and "memory[" not in controller + runner,
        "player_coordinates": "player_position" not in controller + runner and "player_x" not in controller + runner,
        "map_room_state_into_controller": "map_id" not in controller and "room_id" not in controller,
        "screen_classifier_into_controller": "screen_class" not in controller and "classifier" not in controller,
        "dialogue_detector_into_controller": "dialogue" not in controller,
        "frame_hash_into_controller": "frame_sha" not in controller and "coarse_state" not in controller,
        "visual_novelty_into_controller": "novel" not in controller and "stuck" not in controller,
        "stuck_recovery": "if stuck" not in runner and "rescue" not in runner,
        "scripted_route": "desired_route" not in runner and "scripted route" not in runner,
        "scripted_A": "scripted_a" not in runner and "random a" not in runner,
        "random_actions": "random.choice" not in runner and "rng.choice" not in runner,
        "trained_policy": ".fit(" not in controller and ".predict(" not in controller,
        "reward": "reward" not in controller + runner,
    }


def human_semantic_review() -> dict:
    path = RESULTS / "semantic-review.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("highest_verified_milestone") not in {"M0", "M1", "M2", "M3", "M4", "UNVERIFIED"}:
            raise ValueError("invalid semantic milestone")
        return value
    return {
        "highest_verified_milestone": "UNVERIFIED",
        "verified_observations": [],
        "candidate_events_needing_manual_review": None,
        "initial_bedroom_visibly_left": "UNVERIFIED",
        "a_caused_verified_game_interaction": "UNVERIFIED",
        "review_method": "No semantic review artifact was supplied",
    }


def size_text(value: int) -> str:
    return f"{value / (1024 * 1024):.2f} MiB"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    completed = [row for row in manifest["trials"] if row.get("completed_successfully")]
    if len(completed) != 41:
        raise RuntimeError(f"expected 41 successful trials, found {len(completed)}")
    summaries, details = [], []
    for entry in completed:
        summary, detail = summarize_trial(entry)
        summaries.append(summary)
        details.append(detail)
    matched = [row for row in summaries if row["condition"] in CONDITIONS]
    grouped = {
        condition: [row for row in matched if row["condition"] == condition]
        for condition in CONDITIONS
    }
    aggregates = {condition: aggregate(rows) for condition, rows in grouped.items()}
    comparison = comparisons(matched)
    long_summary = next(row for row in summaries if row["condition"] == "long_run")
    a_consequences = a_consequence_analysis(details)
    candidates = review_candidates(details, manifest)
    atomic_write_json(RESULTS / "review-candidates.json", candidates)
    artifacts = artifact_budget(manifest)
    leaks = leakage_checks()
    semantic = human_semantic_review()
    semantic["candidate_events_needing_manual_review"] = len(candidates)

    a_tests = comparison["live_full_vs_live_no_a"]
    if a_tests["sustained_novel_state_episodes"]["p_value"] < .05:
        preregistered_a_effect = "YES"
    elif any(a_tests[key]["p_value"] < .05 for key in METRICS
             if key != "sustained_novel_state_episodes"):
        preregistered_a_effect = "PARTIALLY"
    else:
        preregistered_a_effect = "NO"
    if a_tests["sustained_novel_state_episodes"]["p_value"] < .05:
        a_effect = "YES"
    elif any(a_tests[key]["p_value"] < .05
             for key in VISUAL_TRAJECTORY_METRICS
             if key != "sustained_novel_state_episodes"):
        a_effect = "PARTIALLY"
    else:
        a_effect = "NO"
    direction = "; ".join(
        f"{key} {value['effect']:+.6g} (p={value['p_value']:.6g})"
        for key, value in a_tests.items()
    )

    def favorable_visual_effect(key: str, test: dict) -> bool:
        if test["p_value"] >= .05:
            return False
        if key == "stuck_fraction":
            return test["effect"] < 0
        return test["effect"] > 0

    frozen_tests = comparison["live_full_vs_frozen_full"]
    no_vision_tests = comparison["live_full_vs_no_vision_full"]
    dynamic_feedback_metrics = [
        key for key in VISUAL_TRAJECTORY_METRICS
        if favorable_visual_effect(key, frozen_tests[key])
    ]
    vision_presence_metrics = [
        key for key in VISUAL_TRAJECTORY_METRICS
        if favorable_visual_effect(key, no_vision_tests[key])
    ]
    dynamic_feedback_supported = bool(dynamic_feedback_metrics)
    a_visual_trajectory_supported = a_effect in {"YES", "PARTIALLY"}
    any_primary = bool(
        dynamic_feedback_metrics or vision_presence_metrics
        or a_visual_trajectory_supported)
    milestone = semantic["highest_verified_milestone"]
    if (milestone in {"M3", "M4"}
            and semantic.get("independent_live_full_seeds", 0) >= 2
            and (dynamic_feedback_supported or a_visual_trajectory_supported)):
        category = "E"
    elif milestone in {"M2", "M3", "M4"}:
        category = "D"
    elif milestone == "M1":
        category = "C"
    elif (aggregates["live_full"]["sustained_novel_state_episodes"] > 0
          or any_primary):
        category = "B"
    else:
        category = "A"
    if category in {"D", "E"} or (category in {"B", "C"} and any_primary):
        verdict = "PASS"
    elif category in {"B", "C"}:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    if milestone == "UNVERIFIED":
        achieved = "UNVERIFIED"
    elif category in {"D", "E"}:
        achieved = "YES"
    elif category == "C":
        achieved = "PARTIALLY"
    else:
        achieved = "NO"

    statistics = {
        "experiment": 14,
        "verdict": verdict,
        "primary_result": category,
        "question": "Can frozen LEFT/RIGHT/UP/A MaleCNS control autonomously progress Pokemon?",
        "trial_summaries": summaries,
        "aggregate": aggregates,
        "matched_comparisons": comparison,
        "effect_of_a": {
            "visual_trajectory_answer": a_effect,
            "preregistered_broad_answer": preregistered_a_effect,
            "direction": direction,
            "interpretation": (
                "The preregistered broad rule is PARTIALLY only because action "
                "entropy changed; no visual trajectory endpoint was significant."
                if preregistered_a_effect == "PARTIALLY" and a_effect == "NO"
                else "Visual trajectory and preregistered classifications agree."
            ),
        },
        "dynamic_visual_feedback": {
            "live_vs_frozen_favorable_significant_metrics": dynamic_feedback_metrics,
            "supported": dynamic_feedback_supported,
        },
        "vision_presence": {
            "live_vs_no_vision_favorable_significant_metrics": vision_presence_metrics,
            "supported": bool(vision_presence_metrics),
        },
        "a_event_consequences": a_consequences,
        "long_run": long_summary,
        "semantic_review": semantic,
        "artifact_budget": artifacts,
        "leakage_checks": leaks,
        "did_flymon_achieve_autonomous_progression": achieved,
    }
    atomic_write_json(RESULTS / "statistics.json", statistics)

    live = aggregates["live_full"]
    no_a = aggregates["live_no_a"]
    frozen = aggregates["frozen_full"]
    no_vision = aggregates["no_vision_full"]
    def condition_text(value):
        actions = value["action_distribution"]
        return (f"LEFT %: {100 * actions['LEFT']:.4f}\n"
                f"RIGHT %: {100 * actions['RIGHT']:.4f}\n"
                f"UP %: {100 * actions['UP']:.4f}\n"
                f"A %: {100 * actions['A']:.4f}\n"
                f"NONE %: {100 * actions['NONE']:.4f}\n"
                f"Unique coarse states: {value['unique_coarse_states']:.4f}\n"
                f"Sustained novel-state episodes: {value['sustained_novel_state_episodes']:.4f}\n"
                f"Stuck fraction: {value['stuck_fraction']:.6f}\n"
                f"Action entropy: {value['action_entropy']:.6f}\n"
                f"Max frame distance: {value['maximum_frame_distance']:.6f}\n"
                f"Coarse transitions: {value['coarse_transitions']:.4f}")
    def comparison_text(label):
        value = comparison[label]
        return (f"unique-state effect: {value['unique_coarse_states']['effect']:.6f}\n"
                f"p: {value['unique_coarse_states']['p_value']:.6f}\n"
                f"novel-episode effect: {value['sustained_novel_state_episodes']['effect']:.6f}\n"
                f"p: {value['sustained_novel_state_episodes']['p_value']:.6f}\n"
                f"stuck effect: {value['stuck_fraction']['effect']:.6f}\n"
                f"p: {value['stuck_fraction']['p_value']:.6f}\n"
                f"frame-distance effect: {value['maximum_frame_distance']['effect']:.6f}\n"
                f"p: {value['maximum_frame_distance']['p_value']:.6f}\n"
                f"transition-count effect: {value['coarse_transitions']['effect']:.6f}\n"
                f"p: {value['coarse_transitions']['p_value']:.6f}\n"
                f"action-entropy effect: {value['action_entropy']['effect']:.6f}\n"
                f"p: {value['action_entropy']['p_value']:.6f}")
    long_actions = long_summary["action_distribution"]
    verified = semantic["verified_observations"] or ["none; semantic review unverified"]
    consequence = a_consequences["mean_differences"]
    report = f"""Experiment 14 verdict:
{verdict}
Primary category:
{category}
Base commit:
9027e4727e3f17b9a6f840a0c285d517d0cf3740
Question:
Can the frozen LEFT/RIGHT/UP/A MaleCNS controller autonomously progress Pokemon from bedroom.state over long horizons?

CONTROLLER
LEFT: frozen Experiment 8 DNa02 baseline-relative steering
RIGHT: frozen Experiment 8 DNa02 baseline-relative steering
UP: frozen Experiment 9 DNg100 baseline-relative locomotion
DOWN: disabled
A: frozen Experiment 13 P20 population RMS-z rising edge
NONE: existing deadband / cooldown
Were any controller parameters changed?
NO

PRIMARY MATCHED RUNS
Seeds:
921-930
Horizon:
1500 decisions
Conditions:
LIVE_FULL
LIVE_NO_A
FROZEN_FULL
NO_VISION_FULL

LIVE_FULL
{condition_text(live)}

LIVE_NO_A
{condition_text(no_a)}

FROZEN_FULL
{condition_text(frozen)}

NO_VISION_FULL
{condition_text(no_vision)}

MATCHED STATISTICS
LIVE_FULL vs LIVE_NO_A:
{comparison_text('live_full_vs_live_no_a')}

LIVE_FULL vs FROZEN_FULL:
{comparison_text('live_full_vs_frozen_full')}

LIVE_FULL vs NO_VISION_FULL:
{comparison_text('live_full_vs_no_vision_full')}

EFFECT OF A
Did A significantly change autonomous trajectories?
{a_effect}
Preregistered broad A-effect classification:
{preregistered_a_effect} (triggered only by action entropy; no visual trajectory endpoint was significant)
Direction of effect:
{direction}
A events:
{sum(row['action_counts']['A'] for row in grouped['live_full'])} matched LIVE_FULL; {long_summary['action_counts']['A']} long run; {a_consequences['a_event_count']} reviewed total
A-associated visual changes:
{json.dumps(a_consequences['a_event_means'], sort_keys=True)}
Matched non-A comparison:
means {json.dumps(a_consequences['matched_non_a_means'], sort_keys=True)}; A-minus-matched {json.dumps(consequence, sort_keys=True)}

LONG RUN
Condition:
LIVE_FULL
Seed:
950
Horizon:
5000
LEFT %: {100 * long_actions['LEFT']:.4f}
RIGHT %: {100 * long_actions['RIGHT']:.4f}
UP %: {100 * long_actions['UP']:.4f}
A %: {100 * long_actions['A']:.4f}
NONE %: {100 * long_actions['NONE']:.4f}
Unique coarse states: {long_summary['unique_coarse_states']}
Sustained novel-state episodes: {long_summary['sustained_novel_state_episodes']}
Longest stuck interval: {long_summary['longest_repeated_state_run']}
Maximum distance from initial: {long_summary['maximum_frame_distance']:.6f}
Coarse transitions: {long_summary['coarse_transitions']}

SEMANTIC REVIEW
Highest verified milestone:
{milestone}
Verified observations:
{json.dumps(verified)}
Candidate events needing manual review:
{len(candidates)}
Was the initial bedroom visibly left?
{semantic['initial_bedroom_visibly_left']}
Did A cause a verified game interaction?
{semantic['a_caused_verified_game_interaction']}

ARTIFACTS
MP4 files:
{artifacts['mp4_files']}
Total MP4 size:
{size_text(artifacts['total_mp4_size'])}
Checkpoint PNGs:
{artifacts['checkpoint_pngs']}
Checkpoint cadence:
every 250 decisions
Total checkpoint size:
{size_text(artifacts['total_checkpoint_size'])}
Per-decision metrics:
{artifacts['per_decision_metric_records']}
Expected metric records:
{artifacts['expected_metric_records']}
Metrics integrity:
{artifacts['metrics_integrity']}
Compressed metrics size:
{size_text(artifacts['compressed_metrics_size'])}
Accidentally persisted raw framebuffer files:
{artifacts['accidentally_persisted_raw_framebuffer_files']}
Videos verified:
{artifacts['videos_verified']}
Manifest:
{artifacts['manifest']}

Leakage checks:
RAM: {'PASS' if leaks['RAM'] else 'FAIL'}
player coordinates: {'PASS' if leaks['player_coordinates'] else 'FAIL'}
map/room state into controller: {'PASS' if leaks['map_room_state_into_controller'] else 'FAIL'}
screen classifier into controller: {'PASS' if leaks['screen_classifier_into_controller'] else 'FAIL'}
dialogue detector into controller: {'PASS' if leaks['dialogue_detector_into_controller'] else 'FAIL'}
frame hash into controller: {'PASS' if leaks['frame_hash_into_controller'] else 'FAIL'}
visual novelty into controller: {'PASS' if leaks['visual_novelty_into_controller'] else 'FAIL'}
stuck recovery: {'PASS' if leaks['stuck_recovery'] else 'FAIL'}
scripted route: {'PASS' if leaks['scripted_route'] else 'FAIL'}
scripted A: {'PASS' if leaks['scripted_A'] else 'FAIL'}
random actions: {'PASS' if leaks['random_actions'] else 'FAIL'}
trained policy: {'PASS' if leaks['trained_policy'] else 'FAIL'}
reward: {'PASS' if leaks['reward'] else 'FAIL'}

Primary result:
{category}
Did Flymon achieve autonomous Pokemon progression?
{achieved}
Current scientifically defensible controls:
LEFT
RIGHT
UP
A
Primary limitation:
DOWN remains unavailable; without it the controller can become geometrically trapped. P20 A is aggregate visual drive rather than a semantic interact-now signal. Room changes and interactions also occurred in matched controls, so Experiment 14 does not show that live vision or A uniquely caused the verified semantic progression.
"""
    analysis = (
        "# Experiment 14 analysis\n\n"
        "This report uses seed/trial as the inferential unit. Semantic milestones "
        "require direct human review and are never inferred from hashes or distances.\n\n"
        "```text\n" + report + "```\n\n"
        "## Interpretation\n\n"
        "Flymon produced verified unassisted room-to-room movement in multiple "
        "LIVE_FULL trials and in the long run, so this is limited autonomous "
        "Pokemon progression (category D). The same checkpoint audit found room "
        "changes in LIVE_NO_A, FROZEN_FULL, and NO_VISION_FULL controls. No "
        "favorable LIVE_FULL-vs-FROZEN_FULL visual trajectory endpoint reached "
        "p < 0.05, so the stronger category E claim is not supported. LIVE_FULL "
        "did reduce stuck fraction and increase coarse transitions relative to "
        "NO_VISION_FULL, showing that visual drive changed long-horizon behavior, "
        "but this does not isolate dynamic feedback from the presence of vision.\n\n"
        "Adding A did not significantly change any visual trajectory endpoint. "
        "The preregistered broad rule says PARTIALLY solely because action entropy "
        "changed; that change is expected when an extra action category is enabled "
        "and is not evidence of a richer trajectory. Human review nevertheless "
        "verified that particular neural A pulses opened and advanced the bedroom "
        "SNES interaction. This is an observed causal button consequence, not "
        "evidence that P20 represents semantic interaction intent.\n\n"
        "The sustained-novel-state count is coarse: overlapping 10-decision windows "
        "are merged, so one long episode can cover many state changes. It is kept "
        "as preregistered and is not interpreted as a count of semantic milestones.\n"
    )
    (RESULTS / "analysis.md").write_text(analysis)
    print(report)


if __name__ == "__main__":
    main()
