# Fixed-target feature controls

`src.feature_control_sweep.run_feature_control_sweep` compares two variants of
the current CE-inner / CE-outer optimizer, not a reproduction of GC-SNTK.

- B (`feature_control='assignment'`): the feature-only MLP and cell embeddings
  change soft assignments. Centers are assignment-weighted feature means.
- C (`feature_control='direct'`): Adam optimizes every center coordinate freely.
  There is no convex-hull, nonnegativity or assignment constraint on these centers.

Both freeze the initial soft labels and initial loss weights. The original
joint feature/label/mass optimizer remains the default (`'joint'`).

## Matched initialization and objective

The existing risk partition supplies hard assignments. The existing 0.05
uniform mixing makes the initial soft assignment P0. The MLP output starts at
zero, so its correction to the initial logits is zero. C starts from exactly
the centers of this same P0, not the hard partition means. C loads B's saved
teacher logits and hard partitions after checking their configuration and data.

For z = RMS-normalized S²X, fixed teacher probabilities q = softmax(logits/T),
and m0_j = sum_i P0_ij:

    c0_j = sum_i P0_ij z_i / m0_j
    s0_j = sum_i P0_ij q_i / m0_j
    pi0_j = m0_j / N

Both optimize F(W*(C)) with

    L(W,C) = sum_j pi0_j CE(W [c_j;1], s0_j) + penalty/2 ||W||²
    F(W)   = mean_i CE(W [z_i;1], q_i).

Bias is regularized as in the existing solver. B divides feature sums by the
CURRENT assignment mass to obtain centers, but uses INITIAL mass in both inner
loss and final GCN loss. Fixed loss mass does not impose a column-sum constraint
on the geometric assignment. Autograd includes the center denominator derivative;
it has no path through changes in labels or loss weights. C uses the same
implicit head/adjoint solver and initial-CE gradient scaling. Tracking is inexact
between exact refreshes, as documented in `ce_tracking.md`.

Inverse RMS transformation returns centers to S²X coordinates for fresh
two-layer GCN evaluation with condensed adjacency I. Thus C optimizes synthetic
features in propagated-feature coordinates, not raw X or an adjacency matrix.
Both evaluation losses use pi0. Neither method updates the teacher.

## Sweep and interpretation

Gamma, T, inner penalty and optimization learning rate form a shared full grid.
`assignment_lr` names the optimization rate for both B and C in the existing
runner. Each method independently selects gamma/T/penalty/rate/checkpoint/dropout
by mean search-seed GCN validation. Step zero competes with all later checkpoints.
Teacher validation is diagnostic, not an additional selection stage. Only the
selected configuration of each method gets final-seed test evaluation.

The two selected configurations may have different fixed labels/masses when
gamma or T differs. Therefore `matched_grid.csv` also compares the same gamma,
T, penalty, learning rate, checkpoint and dropout across methods, without test
selection. This separates a matched-setting comparison from independently tuned
performance. One learning-rate grid is equal search budget, not proof of equal
optimizer difficulty. Expand boundaries if needed.

Outputs include each method's full grid, boundary profile, selection, per-seed
scores, checkpoints and optimizer resume state. The comparison directory holds
`summary.csv`, `matched_grid.csv`, `final_deltas.csv` (C minus B, paired by final
student seed), and `initialization_checks.csv`. `latest_comparison.json` points
to the last completed comparison. Rerun identical settings and revision to resume
from saved checkpoints and cached student evaluations. Changing the grid creates
a new fingerprint. Historical A results are context, not rerun matched controls.

The Colab cell runs derivative, frozen-target, common-initialization and resume
tests before training. Local checks are limited to parsing and diff review.
