# Experiment 7 analysis

**Verdict: WEAK. Primary result: B. Answer: PARTIALLY.**

This is an engineered directional BCI, not a natural Pokémon motor circuit or evidence of semantic understanding.

## Criteria

- stable_directional_stream: PASS
- significant_visual_dependence: FAIL
- predictable_left_right_manipulation: PASS
- measurable_closed_loop_feedback: PASS

## Action summaries

```json
{
  "live_vision": {
    "trials": 20,
    "actions_per_trial_mean": 1.65,
    "distribution": {
      "LEFT": 0.00125,
      "RIGHT": 0.00125,
      "UP": 0.0175,
      "DOWN": 0.02125,
      "NONE": 0.95875
    },
    "action_entropy_bits": 0.3025882899187095
  },
  "no_vision": {
    "trials": 20,
    "actions_per_trial_mean": 0.95,
    "distribution": {
      "LEFT": 0.0,
      "RIGHT": 0.0,
      "UP": 0.01125,
      "DOWN": 0.0125,
      "NONE": 0.97625
    },
    "action_entropy_bits": 0.18570968921129943
  },
  "frozen_frame": {
    "trials": 20,
    "actions_per_trial_mean": 1.5,
    "distribution": {
      "LEFT": 0.0025,
      "RIGHT": 0.00125,
      "UP": 0.0175,
      "DOWN": 0.01625,
      "NONE": 0.9625
    },
    "action_entropy_bits": 0.28545749647099505
  },
  "shuffled_vision": {
    "trials": 20,
    "actions_per_trial_mean": 1.3,
    "distribution": {
      "LEFT": 0.0,
      "RIGHT": 0.0,
      "UP": 0.02375,
      "DOWN": 0.00875,
      "NONE": 0.9675
    },
    "action_entropy_bits": 0.23408996669271853
  },
  "controller_null": {
    "trials": 20,
    "actions_per_trial_mean": 0.0,
    "distribution": {
      "LEFT": 0.0,
      "RIGHT": 0.0,
      "UP": 0.0,
      "DOWN": 0.0,
      "NONE": 1.0
    },
    "action_entropy_bits": -0.0
  }
}
```

## Causal results

```json
{
  "visual_dependence": {
    "live_vs_no_vision": {
      "statistic": "total variation distance",
      "observed": 0.01749999999999998,
      "permutations": 9999,
      "p_value": 0.0628
    },
    "jensen_shannon_bits": 0.0026286814226751396,
    "live_vs_frozen": {
      "statistic": "total variation distance",
      "observed": 0.005000000000000016,
      "permutations": 9999,
      "p_value": 0.5328
    },
    "live_frozen_js_bits": 0.0003970019781552009
  },
  "left_right": {
    "left_distribution": {
      "LEFT": 0.00875,
      "RIGHT": 0.0,
      "UP": 0.01875,
      "DOWN": 0.0075,
      "NONE": 0.965
    },
    "right_distribution": {
      "LEFT": 0.0,
      "RIGHT": 0.0025,
      "UP": 0.0225,
      "DOWN": 0.015,
      "NONE": 0.96
    },
    "left_LEFT_RIGHT_ratio": 15.0,
    "right_LEFT_RIGHT_ratio": 0.2,
    "paired_seed_sign_permutation": {
      "observed_mean_signed_bias_difference": 0.01125,
      "p_value": 0.0083
    }
  },
  "closed_loop": {
    "post_action_frame_hash_divergence_fraction": 0.17692307692307693,
    "future_action_disagreement_fraction": 0.021794871794871794,
    "future_candidate_DN_rate_L2_mean_hz": 1.899247372943485
  },
  "interventions": {
    "zero_DNa02": {
      "distribution": {
        "LEFT": 0.0,
        "RIGHT": 0.0,
        "UP": 0.01,
        "DOWN": 0.0225,
        "NONE": 0.9675
      },
      "baseline_distribution": {
        "LEFT": 0.0025,
        "RIGHT": 0.0025,
        "UP": 0.01,
        "DOWN": 0.0225,
        "NONE": 0.9625
      },
      "total_variation": 0.005000000000000003
    },
    "zero_DNg100": {
      "distribution": {
        "LEFT": 0.0025,
        "RIGHT": 0.0025,
        "UP": 0.0,
        "DOWN": 0.0325,
        "NONE": 0.9625
      },
      "baseline_distribution": {
        "LEFT": 0.0025,
        "RIGHT": 0.0025,
        "UP": 0.01,
        "DOWN": 0.0225,
        "NONE": 0.9625
      },
      "total_variation": 0.010000000000000002
    },
    "zero_MDN": {
      "distribution": {
        "LEFT": 0.0025,
        "RIGHT": 0.0,
        "UP": 0.015,
        "DOWN": 0.0,
        "NONE": 0.9825
      },
      "baseline_distribution": {
        "LEFT": 0.0025,
        "RIGHT": 0.0025,
        "UP": 0.01,
        "DOWN": 0.0225,
        "NONE": 0.9625
      },
      "total_variation": 0.02500000000000001
    }
  }
}
```

## Leakage audit

- controller_framebuffer_input: PASS
- controller_RAM_input: PASS
- controller_class_label_input: PASS
- no_trained_policy: PASS
- no_scripted_route: PASS
- no_desired_action_labels: PASS

## Fixed motor versus visual-informative DNs

```json
{
  "identified_motor_DNa02": {
    "left_stimulus_hz": 3.675,
    "right_stimulus_hz": 2.9125,
    "absolute_contrast_hz": 0.7624999999999997
  },
  "frozen_E4_informative_DNs": {
    "left_stimulus_hz": 53.725,
    "right_stimulus_hz": 59.65,
    "absolute_contrast_hz": 5.924999999999997
  },
  "interpretation": "descriptive fixed-population comparison; neither population is trained here"
}
```
