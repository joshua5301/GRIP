# Minimal local-moment variant of the original risk partitioner

The original `method='risk'` remains the default. New `method='risk_local'`
sets `risk_partition(objective_mode='local')` and changes only how cell cross
moments are penalized. Define, in the existing centered RMS-normalized space,

    E_j = (1/N) sum_{i in C_j} (x_i-c_j)(q_i-y_j)^T
    V = (1/N) sum_i ||x_i-c_assignment(i)||^2.

The objectives are

    J_global = B^2/4 * V + 2B * ||sum_j E_j||_F
    J_local  = B^2/4 * V + 2B * sum_j ||E_j||_F.

This is a cellwise Frobenius norm, not a sum of per-class column norms. The
triangle inequality makes J_local >= J_global for any fixed partition. It
prevents cancellation between cells, preserving the original relative coefficient
parameter B. This is not the graph/feature moment-matching optimizer and does not
learn X or edges, introduce probes, or match full feature second moments.

The same kernel teacher, gamma/T/basis, S2X features, joint feature-label D2
surrogate initialization, seed and RNG order, mean representatives, label means,
block proposal/recursive acceptance and stopping criteria are used. There is no
new k-means initialization. Each shared hyperparameter setting starts from the
same partition in both modes. Accepted moves decrease the selected objective;
the first eligible local move deltas are checked against direct recomputation.

Local mode stores cellwise sums of x q^T. Removing a node from cell a changes
E_a by -n_a/[N(n_a-1)] (x-c_a)(q-y_a)^T; adding to b changes E_b by
+n_b/[N(n_b+1)] (x-c_b)(q-y_b)^T. Candidate changes in their Frobenius norms
are computed analytically; batch acceptance recomputes the complete objective.
Cross statistics are accumulated in blocks, avoiding an NxdxC tensor. Storage
is O(KdC), plus candidate-block intermediates. Local mode can cost more than
the global moment calculation. It currently uses the original surrogate seeding,
not the separate split initializer.

For the historical performance comparison, use the same two-layer GCN with
uniform soft CE and identity synthetic adjacency as the old risk experiments.
This deliberately preserves the evaluation protocol: it is not a nonlinear-GCN
risk certificate, and uniform CE differs from the mass-weighted theoretical
setting. The higher-order Taylor remainder is not controlled by this variant.

The additional result fields are local_moment_error, global_moment_error,
cancellation_ratio = global/local, variance_term and moment_term. Zero moments
give ratio 1 by convention. J values across modes and B values are not directly
comparable evidence of predictive quality.

Use the same grid for both methods; compare matched-setting validation scores
as well as independently validation-selected final scores. Historical Optuna
best settings need not be reproduced by a coarse grid. Include B=1.799556227903401,
gamma=.01, T=1, relu and dropout=.9 to cover the reported Cora .026 reference.
Independent final student seeds do not measure partition-seed variability.

No local training or smoke tests were run. Colab verification:

    python -m pytest -q tests/test_risk_local.py tests/test_risk_split.py

Tests cover all legal analytic deltas versus brute force, cancellation across
cells, identical initialization/RNG, exact objectives and mean representatives,
monotonic acceptance, degenerate budgets, and uniform-CE runner integration.
