# Normalized variance sum

For centered propagated features x and teacher probabilities q, freeze global
variances Gx and Gq before partitioning. Minimize Vx/Gx + alpha Vq/Gq through
Var-Part initialization and Lloyd assignment/mean updates in the joint space
[x/sqrt(Gx), sqrt(alpha)(q-mean(q))/sqrt(Gq)]. Constant spaces contribute zero.
Representatives restore the propagated feature scale; labels remain cell means.

Use `method="moment_lloyd_normalized_variance"`, `seeding="bound_var"`.
The existing `space["lambda"]` argument stores alpha for API compatibility;
artifacts explicitly record alpha and the frozen global variances. This objective
is distinct from the bound-derived variance sum and its saturating coefficient.

`teacher_selection="accuracy_only"` selects gamma by teacher validation accuracy,
breaking ties by uncalibrated validation CE and then gamma. T and alpha are
subsequently selected by condensed-student validation; test is evaluated only
for the selected candidate. Student CE remains uniform for previous comparisons.

Partitions and student evaluations are cached under the hashed output directory.
Interrupted individual partitions restart; completed artifacts are reused.

`src.normalized_variance_ablation.run_ablation` holds selected gamma, T, alpha
and student settings fixed across seven variants: main, alpha=0, initialization
only, feature-only Var-Part initialization, unnormalized label variance,
converged k-means++ (one seed), and size-weighted student CE. The latter uses
exactly the main partition. All variants retain mean teacher labels and A=I.
Initialization and normalization interventions affect both seeding and updates
unless specified otherwise. Each variant is evaluated using paired student seeds;
test is descriptive, not a tuning criterion. Common J evaluates every partition
using the original main normalized objective, since native objectives differ.
