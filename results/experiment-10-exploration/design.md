# Experiment 10 — frozen autonomous exploration protocol

Written before any Experiment 10 run. Experiments 1–9 remain unchanged.

The primary controller freezes Experiment 8 DNa02 LEFT/RIGHT and Experiment 9
DNg100 UP. DOWN is absent. The secondary controller adds the exact Experiment 9
MDN DOWN channel, labeled BCI-causal but not naturally validated. All baseline
windows, SD multipliers/floors, 10 neural steps, 8 hold frames, 4 release frames,
one-decision cooldown, summed motor populations, and normalized-excess arbitration
are unchanged. Tie order is LEFT, RIGHT, UP, DOWN. No exploration metric enters
the controller.

Primary seeds 601–620 each run 300 fixed decisions from `states/bedroom.state`
under live current vision, frozen initial vision, and no injection. Seeds 601–605
receive one additional live repeat for robustness. Secondary seeds 621–640 run
300 live decisions with MDN DOWN enabled. Primary seed 650 runs 1,000 live
decisions as a showcase after matched primary trials. Every trial resets PyBoy,
MaleCNS, encoder, controller, and estimates ten no-vision baseline windows.

The unchanged Experiment 3 18×20 LC10a/LPLC2 encoder processes the current
framebuffer. Exact frame and encoder SHA-256 values are recorded. Coarse visual
state is the SHA-256 of the 18×20 grayscale mean-pooled image quantized to 16
uniform levels. Frame distance is RGB mean absolute difference divided by 255;
encoder distance is RMS voltage difference. A visual scene-transition annotation
requires coarse-frame MAE from the previous decision >0.20. This fixed threshold
cannot trigger actions, resets, or early stopping.

Analysis defines a stuck decision as part of a consecutive run of at least ten
identical coarse-state hashes. It also reports longest exact coarse run and
alternating two-state loops. Representative contact sheets use the initial,
maximum-distance, transition, and evenly spaced recorded frames. The horizon never
depends on novelty or apparent progress.

Trial is the statistical unit. Primary live/frozen and live/no comparisons use
paired seed mean differences for unique coarse states, unique encoder hashes,
final frame distance, maximum frame distance, stuck fraction, and action entropy.
Two-sided paired sign permutation tests enumerate all 2^20 signs when practical.
PASS requires live feedback to significantly change exploration relative to a
control and sustained motor output without hidden policies. Sparse/repetitive or
weak feedback is WEAK; rare movement or control equivalence is FAIL. Navigation
events are observations only and never define success.
