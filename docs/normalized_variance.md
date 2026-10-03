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
