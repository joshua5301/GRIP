# Cora 1.3% historical reproduction

This diagnostic loads the original risk partitioner and teacher from commit
`12ceec5810e2c709f98f8aa518d401cd04390168` using Git. It does not restore the removed
legacy experiment runners to the main package. Copies of the reference source are
saved with every run.

The recorded 85.0500% result used B=0.6053863613840811, ReLU teacher gamma=0.01,
T=1, 3000 kernel basis nodes, dropout=0.9, student LR=0.01 and weight decay=0.0005.
The replay uses the historical defaults: condensation seed zero, student seeds
100..109, 1000 epochs and validation evaluation every ten epochs. The original
protocol artifact and environment are unavailable, so this is a reproduction of
recorded settings and matching source, not a guarantee of bitwise reproduction.

The native partitioner initializes by distance sampling in concatenated weighted
features and centered teacher labels. It does not use feature-only FAISS k-means.
The replay records the native initial and final J against the reported
0.10858818925302646 and 0.07925720612237395, with 19 sweeps. It evaluates the native
seed-zero condensate through the original student function and the current evaluator.

The controlled comparison crosses two initializations (native risk and current
feature k-means++) with two optimizers (original risk and current low-rank bilevel
CE). Both use the same graph, teacher targets and uniform-CE two-layer GCN evaluation.
All three condensation seeds participate in selection; final student seeds are
disjoint. Each low-rank initialization chooses one common checkpoint using mean
validation accuracy across the nine search seed pairs. Test scores never select it.
B and low-rank rank/penalty are fixed at their previously selected values; this is
an initialization/objective diagnostic, not an equally budgeted method sweep.

Native risk sampling is still executed before substituting k-means assignments so
the risk solver uses the same subsequent RNG state for node traversal in both arms.
The low-rank model applies its existing 0.05 initial mixing. Hard initial and soft
initial scores are reported separately: a shared partition does not imply identical
initial representatives after softening. Risk hard moves and low-rank soft bilevel
updates remain different algorithms.

Outputs include the historical replay, teacher-logit difference, initial/final J,
per-seed scores, validation trajectories, selected checkpoints and resumable states.
Numerical checks and training are intended for Colab only.
