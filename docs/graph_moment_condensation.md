# Joint graph and feature moment condensation

`src.graph_moment_condensation.run_graph_moment_sweep` learns raw synthetic X and
a symmetric weighted graph. The same frozen GCN checkpoint processes both graphs.
There is no teacher-centroid inversion or KL partition term.

## Objective

With normalized adjacency S including self loops, the two-layer GCN readout is

    Z_theta = S ReLU(S X W1 + b1), logits = Z_theta W2 + b2.

Both message-passing operations are included, exactly matching evaluation-mode
`risk_experiment._forward`. Each probe has fixed original-graph center a and RMS
scale r. On both graphs, U = (Z-a)/r. The targets and synthetic moments are

    M1 = U_original.T Q / N, M1_syn = U_syn.T diag(pi) Y
    M2 = U_original.T U_original / N, M2_syn = U_syn.T diag(pi) U_syn.

The loss averages over probe checkpoints:

    ||M1-M1_syn||_F^2 / max(||M1||_F^2, .01)
      + second_weight * ||M2-M2_syn||_F^2 / max(||M2||_F^2, .01).

M1 includes feature means through its class-column sum. M2 is the full second
moment, including off-diagonals. `second_weight=0` retains only M1. Labels and
masses stay fixed at cell-mean teacher labels and n_j/N, preserving label mass
exactly. No hard-label override or mixture-label update is applied.

The head is affine in Z despite its nonlinear graph encoder. Its label term
depends exactly on M1, while a Taylor expansion of log-sum-exp motivates M2.
Class-weighted second moments are unnecessary for this affine head. This differs
from local clustering under a general nonlinear function of a fixed vector.
The normalized squared loss is an empirical surrogate, not a certified bound:
higher-order residuals remain and a finite bank does not cover all GNNs. The
saved probe_ce_gap measures actual mean absolute soft-CE gap on matching probes,
not held-out students. Neither CE gap nor accuracy must decrease monotonically.

## Protocol

- A GCN teacher trains on original training ground truth with validation
  checkpoint selection. Predictions label all nodes. teacher_run=None prepares
  and caches one; an existing prepare_metric_teacher folder can be reused.
- Probe GCNs use the evaluator's architecture and width. Each trains on K random
  nodes from the full graph using teacher T=1 soft labels. Defaults: seeds
  5000/5001, snapshots 0/50/200 epochs, dropout .5, lr .01, decay .0005. Ground-
  truth validation/test labels are not used in probes or moment optimization.
- Checkpoints are frozen during condensation. Adam cycles through them and
  evaluates the full bank every 25 steps. Lowest bank loss selects the saved
  graph, without validation/test selection. This is not bilevel retraining.
- S2X k-means seed 1234 initializes cells only. X starts at raw cell means and
  is learned directly without convex-hull or nonnegativity constraints.
- The strongest eight quotient neighbors initialize edge weights .8; other
  pairs start at 1e-4. All off-diagonal pairs are learnable. Self loops stay 1;
  symmetric normalization is recomputed. Evaluation applies no threshold. Graph
  storage is KxK, not NxN.
- Evaluation uses the existing two-layer GCN, including its LR and validation
  checkpoint protocol. Learned adjacency is supplied during student training;
  original adjacency is used for evaluation. Default CE is mass weighted.
  Optional uniform CE does not inherit the mass-preserving interpretation.
- Search uses mean validation on seeds 0/1/2; final validation/test uses 100..109
  after all variants are selected. Probe/search/final seed sets are disjoint.

`initial` evaluates the same initial graph, raw means, labels and masses with no
optimization. It is an initialization control, not original GRIP. Each variant
selects its own setting. To isolate M2, compare matched parameters at weight=0
and positive weights rather than only independently selected winners.

## Artifacts and verification

Probe states and original moment targets are cached without retaining full
original embeddings. Condensed graphs are reused across student hyperparameters.
Completed grid trials and final evaluations resume; interrupted fits restart.
Keys include graph/teacher/probe/initialization/grid/optimizer/evaluation settings.
Outputs include protocol.json, probes.pt, condensed_*.pt/.csv, variant trials and
best.json, summary.csv, final_seeds.csv and graph_moments.png. Plots show both
moment errors, matching-probe CE gaps and independent final student accuracy.

The supplied default grid has 72 moment settings, 24 graph optimizations, and 9
initial-control settings: 243 search student trainings and 20 final trainings,
plus cached teacher/probe training.

No local training or smoke tests were run. Run in Colab before the sweep:

    python -m pytest -q tests/test_graph_moment_condensation.py

Tests cover exact readout agreement, sparse/dense graphs, permutation equivariance,
gradients to features and edges, label-location/variance distinction, label-mass
preservation, validation-only selection, adjacency forwarding and resumption.
