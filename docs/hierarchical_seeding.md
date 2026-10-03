# Deterministic hierarchical seeding

The moment Lloyd sweep accepts three deterministic partitions through `seeding`:

| Value | Direction | Split |
|---|---|---|
| `bound_pca` | Existing power-iteration PCA approximation | Mean projection |
| `bound_var` | Coordinate of largest variance | Mean coordinate |
| `bound_pca_sse` | Same PCA approximation | Minimum child SSE in the full augmented space |

All use the same objective-bound feature/label embedding. Each split selects the
cell with largest total scatter. Partitions go directly to filtered updates;
there is no preliminary Lloyd refinement. Set `max_sweeps=0` for initialization only.

The SSE variant sorts projections and evaluates every boundary between distinct
projection values with prefix sums. It maximizes
`||sum_left||^2/n_left + ||sum_right||^2/n_right`, equivalent to minimizing
the sum of child scatters. This is a full-space SSE variant, not an exact
implementation of a projected one-dimensional discriminant threshold method.
Identical projections use a stable median split. Ties follow existing order.
It minimizes this split cost for the chosen direction, not the final moment
objective or student risk. The PCA approximation still uses 100 power iterations.

Suggested Citeseer 0.009 comparison: gamma `[0.05, 0.1, 0.2]`, T `[0.3, 0.5, 0.7]`,
lambda `[10, 30, 50, 100]`, filtered updates up to 100, uniform student CE.
Run each deterministic partition once, search with student seeds 0–4, and evaluate
each validation-selected setting with fresh student seeds 100–109. Report student
standard deviations rather than a fictitious condensation-seed standard deviation.

## Moment versus feature/label variance

`method="moment_lloyd_variance"` uses ordinary Lloyd assignments and mean updates
for `a Vx + b Vq`, with the same fixed bound-derived coefficients and initial
partition as the moment method at matching gamma, T, and lambda. Equivalently it
minimizes `Vx + beta Vq`, where `beta=b/a`. Lambda is a shared parameterization,
not the direct coefficient of label variance. This paired experiment does not
exhaust an independent beta grid. Empty cells are repaired by splitting a donor
cell with more than one member; ties retain old assignments. Convergence means
unchanged assignments; the iteration limit is reported separately.

Artifacts expose both objective values, label variance, and beta. Raw J values
across objectives must not be interpreted as comparable losses. The final GCN
evaluation still uses uniform CE, so the comparison is empirical rather than a
mass-weighted risk guarantee for that downstream student.

## Feature-only Var-Part ablation

`seeding="feature_var"` partitions normalized propagated features only, using the
same Var-Part split rules as `bound_var`. It does not use teacher labels, lambda,
or random seeds to construct the initial assignments. Cell labels are still the
mean teacher probabilities. Subsequent filtered updates optimize the unchanged
`Vx + lambda ||M||` objective. Compare against `bound_var` with the same candidate
grid, teacher features, and student seeds. The `variance` solver mode deliberately
rejects feature-only seeding to avoid silently changing its bound-derived weights.
