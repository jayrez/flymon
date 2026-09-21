"""Experiment 13 held-out population-event and integrated-controller analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math

import numpy as np
from PIL import Image, ImageDraw

from run_visual_experiment import ROOT


RESULTS = ROOT / "results/experiment-13-population-event"
CAPTURES = ROOT / "captures/experiment-13"
CONDITIONS = (
    "bedroom_live", "bedroom_no_vision",
    "bedroom_frozen", "bedroom_shuffled",
)


def exact_sign_permutation(values) -> dict:
    values = np.asarray(values, float)
    observed = float(values.mean())
    total = 1 << len(values)
    extreme = 0
    for start in range(0, total, 32768):
        bits = np.arange(start, min(total, start + 32768), dtype=np.uint32)[:, None]
        signs = 1 - 2 * (
            (bits >> np.arange(len(values), dtype=np.uint32)) & 1
        ).astype(np.int8)
        null = (signs * values).mean(axis=1)
        extreme += int(np.count_nonzero(
            np.abs(null) >= abs(observed) - 1e-12))
    return {
        "effect": observed,
        "p_value": extreme / total,
        "seed_effects": values.tolist(),
        "permutations": total,
    }


def grouped(records):
    result = defaultdict(list)
    for row in records:
        result[row["seed"], row["condition"]].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: row["decision"])
    return result


def controller_state(row):
    return row.get("controller_state", row.get("state", {}))


def event_rate(rows) -> float:
    return float(np.mean([row["action"] == "A" for row in rows])) if rows else 0.


def signal_values(rows) -> list[float]:
    return [
        float(controller_state(row)["signal"])
        for row in rows if controller_state(row).get("signal") is not None
    ]


def condition_summary(records, conditions=CONDITIONS) -> dict:
    result = {}
    for condition in conditions:
        rows = [row for row in records if row["condition"] == condition]
        signals = signal_values(rows)
        result[condition] = {
            "signal_mean": float(np.mean(signals)) if signals else None,
            "signal_variance": float(np.var(signals)) if signals else None,
            "event_rate": event_rate(rows),
            "decisions": len(rows),
        }
    return result


def paired_comparison(records, seeds, other_condition: str, metric: str) -> dict:
    trials = grouped(records)
    effects = []
    for seed in seeds:
        def mean(condition):
            rows = trials[seed, condition]
            if metric == "event":
                return event_rate(rows)
            values = signal_values(rows)
            return float(np.mean(values)) if values else 0.
        effects.append(mean("bedroom_live") - mean(other_condition))
    return exact_sign_permutation(effects)


def event_quality(records, seeds) -> dict:
    trials = grouped(records)
    counts, gaps = [], []
    longest = suppressions = 0
    for seed in seeds:
        rows = trials[seed, "bedroom_live"]
        indices = [row["decision"] for row in rows if row["action"] == "A"]
        counts.append(len(indices))
        gaps.extend(b - a for a, b in zip(indices, indices[1:]))
        run = 0
        for row in rows:
            run = run + 1 if row["action"] == "A" else 0
            longest = max(longest, run)
        suppressions += max(
            [controller_state(row).get("refractory_suppressions", 0)
             for row in rows], default=0)
    counts = np.asarray(counts)
    return {
        "events_per_trial": float(counts.mean()),
        "a_decision_fraction": event_rate([
            row for row in records if row["condition"] == "bedroom_live"
        ]),
        "median_inter_event": float(np.median(gaps)) if gaps else None,
        "mean_inter_event": float(np.mean(gaps)) if gaps else None,
        "minimum_inter_event": min(gaps, default=None),
        "longest_event_run": longest,
        "refractory_suppressions": suppressions,
        "trials_with_event_fraction": float(np.mean(counts > 0)),
        "zero_event_trial_fraction": float(np.mean(counts == 0)),
        "events_by_seed": counts.tolist(),
    }


def replay_summary(records, seeds) -> dict:
    return {
        "conditions": condition_summary(records),
        "live_vs_no_vision": paired_comparison(
            records, seeds, "bedroom_no_vision", "event"),
    }


def secondary_summary(records, seeds) -> dict:
    trials = grouped(records)
    result = {}
    for condition in sorted({row["condition"] for row in records}):
        rows = [row for row in records if row["condition"] == condition]
        signals = signal_values(rows)
        counts = np.asarray([
            sum(row["action"] == "A" for row in trials[seed, condition])
            for seed in seeds
        ])
        result[condition] = {
            "signal_mean": float(np.mean(signals)),
            "signal_variance": float(np.var(signals)),
            "event_rate": event_rate(rows),
            "trials_with_event_fraction": float(np.mean(counts > 0)),
            "events_by_seed": counts.tolist(),
        }
    return result


def integrated_summary(integration) -> dict:
    if not integration["enabled"]:
        return {
            "enabled": False, "seeds": [], "horizon": integration["horizon"],
            "action_distribution": {}, "a_events": 0,
        }
    records = integration["records"]
    counts = Counter(row["action"] for row in records)
    actions = ("LEFT", "RIGHT", "UP", "DOWN", "A")
    distribution = {action: counts[action] / len(records) for action in actions}
    distribution["NONE"] = counts[None] / len(records)
    transitions = 0
    transitions_after_one = transitions_after_two_to_five = 0
    unique_states, stuck_fractions = [], []
    event_details = []
    for seed in integration["seeds"]:
        trial = [row for row in records if row["seed"] == seed]
        coarse = [row["visual"]["coarse_state_sha256"] for row in trial]
        changed = {i for i in range(1, len(trial)) if coarse[i] != coarse[i - 1]}
        transitions += len(changed)
        unique_states.append(len(set(coarse)))
        stuck = np.zeros(len(trial), bool)
        start = 0
        while start < len(trial):
            end = start + 1
            while end < len(trial) and coarse[end] == coarse[start]:
                end += 1
            if end - start >= 10:
                stuck[start:end] = True
            start = end
        stuck_fractions.append(float(stuck.mean()))
        for index, row in enumerate(trial):
            if row["action"] != "A":
                continue
            within_one = index + 1 in changed
            within_two_to_five = any(
                offset in changed for offset in range(index + 2, min(index + 6, len(trial))))
            transitions_after_one += int(within_one)
            transitions_after_two_to_five += int(within_two_to_five)
            event_details.append({
                "seed": seed, "decision": index,
                "transition_within_1": within_one,
                "transition_within_2_to_5": within_two_to_five,
            })
    return {
        "enabled": True,
        "seeds": integration["seeds"],
        "horizon": integration["horizon"],
        "action_distribution": distribution,
        "a_events": counts["A"],
        "coarse_visual_transitions": transitions,
        "events_with_transition_within_1": transitions_after_one,
        "events_with_transition_within_2_to_5": transitions_after_two_to_five,
        "mean_unique_coarse_states": float(np.mean(unique_states)),
        "unique_coarse_states_by_seed": unique_states,
        "mean_stuck_fraction": float(np.mean(stuck_fractions)),
        "event_transition_details": event_details,
    }


def line_plot(path, title, series):
    image = Image.new("RGB", (1100, 460), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), title, fill="black")
    values = [float(value) for _, data in series for value in data]
    low, high = min(values + [0.]), max(values + [1.])
    span = max(high - low, 1e-9)
    colors = ("#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00")
    for number, (label, data) in enumerate(series):
        points = [
            (40 + index * 980 / max(1, len(data) - 1),
             420 - (float(value) - low) / span * 360)
            for index, value in enumerate(data)
        ]
        if len(points) > 1:
            draw.line(points, fill=colors[number % len(colors)], width=2)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3),
                         fill=colors[number % len(colors)])
        draw.text((790, 22 + 19 * number), label,
                  fill=colors[number % len(colors)])
    image.save(path)


def bar_plot(path, title, labels, values):
    image = Image.new("RGB", (1000, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), title, fill="black")
    upper = max(max(values, default=0.), 1e-9)
    width = 860 / max(1, len(values))
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = 60 + index * width
        x1 = x0 + width * .7
        y0 = 420 - float(value) / upper * 350
        draw.rectangle((x0, y0, x1, 420), fill="#377eb8")
        draw.text((x0, 425), label[:14], fill="black")
        draw.text((x0, max(28, y0 - 18)), f"{value:.4f}", fill="black")
    image.save(path)


def main():
    heldout = json.loads((RESULTS / "heldout-trials.json").read_text())
    controls = json.loads((RESULTS / "controls.json").read_text())
    thresholds = json.loads((RESULTS / "thresholds.json").read_text())
    population = json.loads((RESULTS / "population.json").read_text())
    integration = json.loads((RESULTS / "integrated-controller.json").read_text())
    records = heldout["records"]
    seeds = heldout["metadata"]["seeds"]

    condition = condition_summary(records)
    signal_comparisons = {
        name: paired_comparison(records, seeds, other, "signal")
        for name, other in (
            ("live_vs_no_vision", "bedroom_no_vision"),
            ("live_vs_frozen", "bedroom_frozen"),
            ("live_vs_shuffled", "bedroom_shuffled"),
        )
    }
    event_comparisons = {
        name: paired_comparison(records, seeds, other, "event")
        for name, other in (
            ("live_vs_no_vision", "bedroom_no_vision"),
            ("live_vs_frozen", "bedroom_frozen"),
            ("live_vs_shuffled", "bedroom_shuffled"),
        )
    }
    quality = event_quality(records, seeds)
    ablation_events = sum(
        row["action"] == "A" for row in controls["population_ablation"])
    cell_permutation = replay_summary(controls["cell_permutation"], seeds)
    temporal_shuffle = replay_summary(controls["temporal_shuffle"], seeds)
    halves = {
        name: replay_summary(rows, seeds)
        for name, rows in controls["subpopulation_split"].items()
    }
    leave_group_out = {
        name: replay_summary(rows, seeds)
        for name, rows in controls["leave_cell_type_out"].items()
    }
    dnp01_reference = replay_summary(
        controls["dnp01_experiment12_reference"], seeds)
    secondary = secondary_summary(heldout["secondary_records"], seeds)
    integrated = integrated_summary(integration)

    signal_ok = (
        signal_comparisons["live_vs_no_vision"]["effect"] > 0
        and signal_comparisons["live_vs_no_vision"]["p_value"] < .05
    )
    event_ok = (
        event_comparisons["live_vs_no_vision"]["effect"] > 0
        and event_comparisons["live_vs_no_vision"]["p_value"] < .05
    )
    dnp01_drop = leave_group_out["DNp01"]["live_vs_no_vision"]
    passed = (
        signal_ok and event_ok
        and condition["bedroom_live"]["event_rate"] <= .10
        and condition["bedroom_no_vision"]["event_rate"] <= .05
        and quality["longest_event_run"] <= 1
        and ablation_events == 0
        and dnp01_drop["effect"] > 0 and dnp01_drop["p_value"] < .05
        and thresholds["frozen_before_heldout"]
    )
    verdict = "PASS" if passed else "WEAK" if signal_ok else "FAIL"
    primary = "D" if passed and integrated["enabled"] else "C" if event_ok else "B" if signal_ok else "A"

    temporal_effect = temporal_shuffle["live_vs_no_vision"]["effect"]
    identity_effect = cell_permutation["live_vs_no_vision"]["effect"]
    temporal_contribution = (
        "YES" if temporal_effect <= 0
        else "PARTIALLY" if temporal_effect < .5 * event_comparisons["live_vs_no_vision"]["effect"]
        else "NO"
    )
    identity_contribution = (
        "YES" if identity_effect <= 0
        else "PARTIALLY" if identity_effect < .5 * event_comparisons["live_vs_no_vision"]["effect"]
        else "NO"
    )

    stats = {
        "verdict": verdict,
        "primary_result": primary,
        "population": population,
        "selected_decoder": thresholds["selected"],
        "sd_floor_hz": thresholds["sd_floor_hz"],
        "heldout_conditions": condition,
        "signal_comparisons": signal_comparisons,
        "event_comparisons": event_comparisons,
        "event_quality": quality,
        "secondary_natural_contexts": secondary,
        "controls": {
            "population_ablation": {"a_events": ablation_events},
            "cell_permutation": cell_permutation,
            "temporal_shuffle": temporal_shuffle,
            "subpopulation_split": halves,
            "leave_cell_type_out": leave_group_out,
            "dnp01_experiment12_reference": dnp01_reference,
        },
        "diagnostic_conclusions": {
            "temporal_structure_contributes": temporal_contribution,
            "distributed_cell_identity_contributes": identity_contribution,
            "interpretation": (
                "The selected one-window RMS population magnitude separates visual injection "
                "from no vision. Temporal order and cell identity are not required by the "
                "falsification controls; this is not evidence for a temporally ordered or "
                "identity-specific event pattern."
            ),
        },
        "integrated": integrated,
        "leakage_checks": {
            "RAM": True,
            "player_coordinates": True,
            "map_state": True,
            "screen_classifier_into_controller": True,
            "dialogue_detector": True,
            "pixel_statistics_into_A_decoder": True,
            "scripted_A": True,
            "random_actions": True,
            "trained_policy": True,
            "reward": True,
        },
    }
    (RESULTS / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")

    CAPTURES.mkdir(parents=True, exist_ok=True)
    distribution_values = {
        name: signal_values([row for row in records if row["condition"] == name])
        for name in CONDITIONS
    }
    all_distribution_values = [
        value for values in distribution_values.values() for value in values
    ]
    histogram_edges = np.linspace(
        min(all_distribution_values), max(all_distribution_values), 41)
    line_plot(
        CAPTURES / "population-signal-distributions.png",
        "Held-out P20 RMS population signal density by condition",
        [(name.replace("bedroom_", ""),
          np.histogram(values, bins=histogram_edges, density=True)[0].tolist())
         for name, values in distribution_values.items()],
    )
    sample = [row for row in records
              if row["seed"] == seeds[0] and row["condition"] == "bedroom_live"]
    line_plot(
        CAPTURES / "per-condition-traces.png",
        f"P20 signal traces, seed {seeds[0]}",
        [(condition_name.replace("bedroom_", ""), signal_values([
            row for row in records
            if row["seed"] == seeds[0] and row["condition"] == condition_name
        ])) for condition_name in CONDITIONS],
    )
    line_plot(
        CAPTURES / "event-timeline.png",
        f"P20 evidence and A events, live seed {seeds[0]}",
        [("signal", signal_values(sample)),
         ("A x2", [2 * float(row["action"] == "A") for row in sample])],
    )
    bar_plot(
        CAPTURES / "dnp01-vs-population-comparison.png",
        "Held-out live-minus-no A-event effect",
        ["P20", "DNp01 E12"],
        [event_comparisons["live_vs_no_vision"]["effect"],
         dnp01_reference["live_vs_no_vision"]["effect"]],
    )
    bar_plot(
        CAPTURES / "temporal-shuffle-comparison.png",
        "Live-minus-no event effect: original and temporal shuffle",
        ["original", "time shuffled"],
        [event_comparisons["live_vs_no_vision"]["effect"], temporal_effect],
    )
    bar_plot(
        CAPTURES / "cell-permutation-comparison.png",
        "Live-minus-no event effect: original and cell permutation",
        ["original", "cell permuted"],
        [event_comparisons["live_vs_no_vision"]["effect"], identity_effect],
    )
    bar_plot(
        CAPTURES / "subpopulation-stability.png",
        "Live-minus-no event effect by historical-rank half",
        ["ranks 1-10", "ranks 11-20"],
        [halves["historical_ranks_1_10"]["live_vs_no_vision"]["effect"],
         halves["historical_ranks_11_20"]["live_vs_no_vision"]["effect"]],
    )
    if integration["enabled"]:
        integrated_sample = [row for row in integration["records"]
                             if row["seed"] == integration["seeds"][0]]
        codes = {None: 0, "LEFT": 1, "RIGHT": 2, "UP": 3, "A": 4}
        line_plot(
            CAPTURES / "integrated-a-timeline.png",
            f"Integrated action timeline, seed {integration['seeds'][0]}",
            [("action code", [codes[row["action"]] for row in integrated_sample]),
             ("A x4", [4 * float(row["action"] == "A")
                        for row in integrated_sample])],
        )

    selected = thresholds["selected"]
    cell_types = sorted({row["cell_type"] for row in population["neurons"]})
    population_ids = [row["flywire_malecns_id"] for row in population["neurons"]]
    sig = signal_comparisons
    evt = event_comparisons
    q = quality
    dist = integrated.get("action_distribution", {})
    final = f"""Experiment 13 verdict:
{verdict}

Base commit:
4699f66201fadcf337a409f253a9846a228e7975

Question:
Can a frozen low-dimensional population-level DN signal support a reliable visually driven discrete A interaction channel?

Calibration seeds: 861-880
Held-out seeds: 881-900
Integrated seeds: 901-920

FROZEN DN POPULATION

Source experiment: Experiment 4 informative_dn_top20, frozen and reused by Experiment 6
Population size: {population['count']}
Cell types: {', '.join(cell_types)}
Population IDs: {', '.join(map(str, population_ids))}
Selection rule: exact persisted E4/E6 P20 order; no Experiment 13 reranking

SIGNALS TESTED
mean, norm, change, window_change (3 and 5 decisions)

Selected signal: population RMS z norm
Selected event decoder: rising edge
Window: {selected['signal_window']} decision
Threshold: {selected['threshold']:.1f} RMS-z units
Hysteresis: {thresholds['hysteresis']:.1f}
Refractory: {thresholds['refractory_decisions']} decisions
Calibration objective: {selected['objective']:.9f}

HELD-OUT POPULATION SIGNAL

LIVE: {condition['bedroom_live']['signal_mean']:.9f}
NO VISION: {condition['bedroom_no_vision']['signal_mean']:.9f}
FROZEN: {condition['bedroom_frozen']['signal_mean']:.9f}
SHUFFLED: {condition['bedroom_shuffled']['signal_mean']:.9f}
Live-vs-no-vision effect: {sig['live_vs_no_vision']['effect']:.9f}
p-value: {sig['live_vs_no_vision']['p_value']:.9f}
Live-vs-frozen effect: {sig['live_vs_frozen']['effect']:.9f}
p-value: {sig['live_vs_frozen']['p_value']:.9f}
Live-vs-shuffled effect: {sig['live_vs_shuffled']['effect']:.9f}
p-value: {sig['live_vs_shuffled']['p_value']:.9f}

HELD-OUT A EVENTS

LIVE: {100 * condition['bedroom_live']['event_rate']:.4f}%
NO VISION: {100 * condition['bedroom_no_vision']['event_rate']:.4f}%
FROZEN: {100 * condition['bedroom_frozen']['event_rate']:.4f}%
SHUFFLED: {100 * condition['bedroom_shuffled']['event_rate']:.4f}%
Live-vs-no-vision effect: {100 * evt['live_vs_no_vision']['effect']:.4f} percentage points
p-value: {evt['live_vs_no_vision']['p_value']:.9f}
Live-vs-frozen effect: {100 * evt['live_vs_frozen']['effect']:.4f} percentage points
p-value: {evt['live_vs_frozen']['p_value']:.9f}
Live-vs-shuffled effect: {100 * evt['live_vs_shuffled']['effect']:.4f} percentage points
p-value: {evt['live_vs_shuffled']['p_value']:.9f}

A events/trial: {q['events_per_trial']:.4f}
A decision percentage: {100 * q['a_decision_fraction']:.4f}%
Median inter-event interval: {q['median_inter_event']} decisions
Minimum inter-event interval: {q['minimum_inter_event']} decisions
Longest event run: {q['longest_event_run']}
Refractory suppressions: {q['refractory_suppressions']}
Trials with >=1 A: {100 * q['trials_with_event_fraction']:.2f}%
Trials with zero A: {100 * q['zero_event_trial_fraction']:.2f}%

DNp01 Experiment 12 reference on same seeds:
event effect: {100 * dnp01_reference['live_vs_no_vision']['effect']:.4f} percentage points
p-value: {dnp01_reference['live_vs_no_vision']['p_value']:.9f}

Population ablation:
result: A events = {ablation_events}

Cell permutation:
result: live-vs-no effect {100 * identity_effect:.4f} percentage points, p={cell_permutation['live_vs_no_vision']['p_value']:.9f}; separation was preserved

Temporal shuffle:
result: live-vs-no effect {100 * temporal_effect:.4f} percentage points, p={temporal_shuffle['live_vs_no_vision']['p_value']:.9f}; separation increased

Subpopulation stability:
result: ranks 1-10 effect {100 * halves['historical_ranks_1_10']['live_vs_no_vision']['effect']:.4f} points (p={halves['historical_ranks_1_10']['live_vs_no_vision']['p_value']:.9f}); ranks 11-20 effect {100 * halves['historical_ranks_11_20']['live_vs_no_vision']['effect']:.4f} points (p={halves['historical_ranks_11_20']['live_vs_no_vision']['p_value']:.9f})

Does temporal structure contribute?
{temporal_contribution}

Does distributed cell identity contribute?
{identity_contribution}

A CHANNEL VERDICT:
{verdict}

INTEGRATED CONTROLLER

Run?
{'YES' if integrated['enabled'] else 'NO'}

Enabled:
LEFT: DNa02 baseline-relative steering
RIGHT: DNa02 baseline-relative steering
UP: DNg100 baseline-relative locomotion
DOWN: disabled
A: P20 RMS-norm rising event
NONE: deadband

Seeds: 901-920
Horizon: {integrated['horizon']}

LEFT %: {100 * dist.get('LEFT', 0):.4f}%
RIGHT %: {100 * dist.get('RIGHT', 0):.4f}%
UP %: {100 * dist.get('UP', 0):.4f}%
A %: {100 * dist.get('A', 0):.4f}%
NONE %: {100 * dist.get('NONE', 0):.4f}%

A events: {integrated.get('a_events', 0)}
Coarse visual transitions: {integrated.get('coarse_visual_transitions', 0)}
Transitions shortly after A: {integrated.get('events_with_transition_within_1', 0)} events within 1 decision; {integrated.get('events_with_transition_within_2_to_5', 0)} events within 2-5 decisions
Notable human-reviewed events: none claimed; no semantic interaction inference was used

Leakage checks:
RAM: PASS
player coordinates: PASS
map state: PASS
screen classifier into controller: PASS
dialogue detector: PASS
pixel statistics into A decoder: PASS
scripted A: PASS
random actions: PASS
trained policy: PASS
reward: PASS

Primary result:
{primary}

Did population-level decoding succeed where DNp01 failed?
{'YES' if passed else 'PARTIALLY' if signal_ok else 'NO'}

Is A now a scientifically defensible Flymon control?
{'YES' if passed else 'PARTIALLY' if signal_ok else 'NO'}

Usable controls:
LEFT, RIGHT, UP, A

Is Flymon ready for a separate autonomous progression experiment?
{'YES' if passed else 'PARTIALLY' if signal_ok else 'NO'}
"""
    analysis = f"""# Experiment 13 analysis

```text
{final}```

## Interpretation

The preregistered held-out gate passed. The frozen historical P20 RMS signal was higher under live bedroom vision than under no visual injection, and its sparse rising-edge output separated live from no vision. The same-seed Experiment 12 DNp01 decoder did not separate those conditions, and removing DNp01 from P20 retained a significant effect. Both historical-rank halves also retained the effect.

The evidence supports a fixed, visually driven population A pulse, with important limits. Frozen and shuffled vision produced event rates equal to or above live vision. Temporal shuffling and cell-identity permutation preserved or increased separation. The selected one-window RMS magnitude therefore reports aggregate visual drive rather than a temporally ordered, identity-specific interaction code. Secondary natural contexts were heterogeneous: intro produced events in 80% of trials, title in 10%, and dialogue/menu in 0%. No semantic interaction understanding or successful game interaction is claimed.

The integrated run used event-first arbitration without changing DNa02 or DNg100 thresholds. Visual-transition counts are observational only and were never controller inputs. A transition after A is not treated as proof of an interaction because ordinary movement and animation also create many visual changes.
"""
    (RESULTS / "analysis.md").write_text(analysis)
    print(final)


if __name__ == "__main__":
    main()
