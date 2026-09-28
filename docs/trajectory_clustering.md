# CE trajectories and alternating clustering

`src.trajectory_clustering.run_trajectory_sweep` implements free representative
optimization with hard assignments. The first experiment reuses the saved Arxiv
909-cell CE-bilevel reference: teacher logits, gamma, RMS S²X representation,
GCN settings and optional risk partition. K-means++ is the default initialization.
Both initializations start at hard cell means, without the old 5% soft mixing.
Changing T changes the fixed teacher probabilities; gamma is held at its saved
value. The risk initialization, if selected, remains the saved partition rather
than being recomputed for each T.

## Objective

For frozen student checkpoints theta_t and teacher probabilities q_i:

    L_it = CE(f_theta_t(z_i), q_i)
    J = mean_(i,t) |L_it - CE(f_theta_t(c_a(i)), q_i)|.

All checkpoints have equal weight. `loss="absolute"` optimizes this J directly.
`loss="squared"` replaces absolute differences by squared differences, producing
J2. Labels are arithmetic cell averages and CE mass is n_j/N. CE linearity in
its target gives mean_t |R_original(theta_t)-R_condensed(theta_t)| <= J, or
<= sqrt(J2). The implementation records both the actual risk gap and its bound.
The per-node replacement cost uses q_i, not the cell-average target s_j.

## Alternation

Each outer round trains new regularized linear SGC students on the current
representatives, their mean labels, and mass CE. The representation is fixed
RMS S²X; the bias is also penalized. Adam trains for a fixed number of epochs;
this is an observed finite training trajectory, not a claimed converged optimum.
The same explicit initialization seeds are reused each round, with fresh heads
and optimizer states, to control initialization variation. Saved checkpoints
include no dropout and are evaluated deterministically. Their final CE and
gradient residuals are recorded in `trajectory_training.csv`.

The latest `retain_rounds` trajectories are frozen as a buffer. For each inner
Lloyd iteration, all node-cell costs are evaluated and each node chooses the
lowest-cost representative. One existing member per cell is anchored at its
previous assignment to preserve the 909 nonempty cells. Anchors are the members
with smallest current replacement cost; ties use node ID. Other ties preserve
the previous assignment. Thus this is a feasible nonempty assignment update,
not an exact solution of the globally constrained nonempty assignment problem.
It never increases the fixed-head objective, up to numerical error.

With assignments frozen, Adam directly optimizes representative features. The
lowest-objective iterate, including the starting features, is retained. Labels
and masses are then recomputed from memberships. This guarantees nonincrease
for each completed inner block, not every Adam proposal. The next outer round
restarts student training. A new trajectory buffer defines a new objective;
objective values across rounds are not a single monotone curve.

Representatives are unconstrained features, not convex combinations or means
after optimization. There are no assignment logits, implicit Hessian solves,
student-training derivatives, or free label parameters. Costs stream node-cell
blocks, and the center gradient streams node-checkpoint-class blocks. Neither
an N-by-N distance matrix nor a full N-by-M-by-K tensor is retained. Float64
computations use the analytic CE derivative through frozen head outputs.

## Evaluation and persistence

The synthetic graph is A=I. Inverting the affine RMS scaling maps the learned
representatives back to S²X feature coordinates for fresh two-layer GCN students;
these students are evaluated on the original graph. SGC-to-GCN transfer remains
an empirical question. The fixed-head risk bound does not guarantee unseen
trajectory preservation, accuracy improvement, or GCN transfer.

The grid searches T, head penalty, representative learning rate and objective
variant. Gamma and GCN training hyperparameters remain those of the reference.
Three GCN search seeds select a candidate and outer round, including round zero,
using validation only. Test labels are evaluated only after this choice, with
disjoint final seeds, for initial, selected and last representatives of the
selected candidate. The summary does not select candidates by test accuracy.

Each completed outer round is saved atomically. Rerunning the same cell resumes
at the last completed round; a partially completed round repeats. GCN evaluation
is cached per seed, and identical round-zero candidates share an evaluation
cache. Configuration and input digests prevent mixing different runs. Code uses
an explicit algorithm version for cache identity, so unrelated Git commits do
not restart this sweep. Different initializations may run in separate sessions;
two sessions must not write to the same configuration folder.

Tests cover streamed value/gradient agreement with direct autograd, exact
mass/label preservation and risk bounds, nonempty assignment descent, center
descent, round-boundary resume, and validation-only selection. Numerical tests
are intended for Colab; only static checks are performed locally.
