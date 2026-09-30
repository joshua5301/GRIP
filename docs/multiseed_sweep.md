# Cora joint condensation-seed grid

Run one density per session: 0.013 (35 cells), 0.026 (70), and 0.052 (140).
Each candidate is a Cartesian combination of T, rank and inner penalty. Assignment
learning rate is fixed. Condensation seeds 0, 1 and 2 determine both feature k-means
and low-rank factors. Each candidate/checkpoint is scored using the mean GCN
validation accuracy over all three condensation seeds and three student seeds.
Choose one common hyperparameter tuple and checkpoint, not separate best settings
or steps per condensation seed. Missing or duplicate seed pairs fail aggregation.

The teacher uses a separate fixed seed and gamma is selected once per density run
by teacher validation accuracy, before condensation. Independent sessions use the
same data, teacher seed, kernel, basis and gamma grid, so they reproduce the same
teacher selection without sharing writable cache files. selected_teacher.json
records gamma and a logit digest to verify agreement between sessions. No teacher
variation is introduced across condensation seeds within a density.

After gamma selection, teacher_calibration.json records a scalar temperature fit
to validation CE over T in [0.01, 100]. It reports boundary status, CE before and
after, and the invariant teacher accuracy. This is diagnostic only: the grid's
inner/outer T values and teacher logits are unchanged. No test labels are used.

Features are RMS-normalized S^2X for optimization, with the existing inverse scale
used for synthetic GCN inputs. Low-rank corrections start from k-means assignments.
Inner linear CE is cell-mass weighted; outer teacher CE is uniform over original
nodes; both targets use the same fixed T. Final GCN training has two layers,
identity synthetic adjacency and uniform CE. Newton-first and implicit warm start
are enabled; grouped PCG and probability caching remain disabled.

After validation selection, evaluate the chosen candidate at step zero and the
selected step for each condensation seed using student seeds 100 through 109.
Report per-condensation-seed means and student standard deviations, the overall
mean, and the sample standard deviation of the three condensation-seed means.
The 30 student fits are not 30 independent condensations. The three condensation
seeds also participate in hyperparameter selection and are not held-out seeds.

Output is separate per ratio/configuration. Every candidate/condensation seed has
its own resumable optimizer state and cached GCN evaluations. Rerunning the same
cell resumes interrupted work. Changing the grid or configuration creates a new
run. A rerun skips completed optimization and evaluation; it recomputes aggregation.
Only selected configurations receive final test evaluation. No local numerical
tests or training are needed; run the provided tests in Colab.

## Prototype and MLP sessions

`run_cora_multiseed(..., method="prototype")` accepts grid keys `T`, `tau`,
`penalty`. It learns cell prototypes in fixed RMS-normalized S²X space. Assignment
probabilities are softmax of negative squared distance divided by `tau`; no
baseline logits or Lloyd iteration are used. Prototypes start at the same k-means
cell means used to define the other methods' initialization. Realized condensed
features and labels remain assignment-weighted means, not the prototypes.

`method="mlp"` accepts `T`, `rank`, `width`, `penalty`. The existing feature-only
encoder is z -> width -> ReLU -> rank, followed by cell embeddings. It learns a
rank-scaled logit correction on top of the 0.05-smoothed k-means assignment.
Its last encoder layer starts at zero, preserving the existing initial assignment.
This is the previous residual MLP family, not a pure MLP softmax without a baseline.

Teacher T and assignment tau are separate. Both use the same mass-weighted inner
CE, original-node outer CE, exact Newton-first solver, warm starts and uniform-CE
GCN evaluation. Shared seeds and initial hard cells do not make their initial soft
assignments identical. Every candidate/seed saves `initial_assignment.json` with
hard-assignment agreement, entropy, mass TV and realized-center RMS difference.
Initial and selected GCN performance are evaluated separately on the final seeds.

Use different output directories for the two sessions. Each selects a common
checkpoint and configuration using all nine search seed pairs; test labels are
used only after selection. Identical reruns resume and reuse cached evaluations.
The original low-rank API and its default run fingerprints remain unchanged.
Prototype numerical-gradient and resume tests are in `tests/test_prototype_assignment.py`;
run them in Colab, not locally.
