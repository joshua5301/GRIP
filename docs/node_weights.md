# Learnable node weights for low-rank assignments

Let P be the row-softmax of the original low-rank-corrected assignment logits.
For new trainable node logits a, define w = N softmax(a) and A_ij = w_i P_ij.
Initialize a at zero. Each row mass may change, but total mass remains N.
The representative moments are A^T [1, Z, Q] / N; features and labels remain
convex averages with identical coefficients, and mass CE uses the resulting
column masses. Columns are not balanced. The raw source logits, RMS features,
initial partition and student settings are reused.

Optimize outer_teacher_CE + lambda KL(w/N || uniform), with both gradient terms
divided by the same fixed initial outer CE scale used by the existing optimizer.
The outer teacher CE remains a uniform mean over all original nodes: w changes
only the condensed training distribution. Lambda zero means no weight penalty,
not fixed weights. Nonnegative lambda and a separate node-logit learning rate are
swept while the source low-rank winner's rank, T, assignment learning rate and
inner regularization remain fixed. These weights are learned importance, not a
calibrated probability of label correctness.

The streamed weighted moment operator recomputes assignment probabilities in
backward. It provides gradients to U, V and w without storing the full assignment
matrix or a weighted copy of the feature material. Extra optimizer parameters
are N scalars. Checkpoints store node logits; logs include weight KL, mean, min,
max and effective sample size fraction 1 / mean(w^2). A fraction of one means
uniform weights; smaller values mean greater concentration.

`run_node_weight_sweep` shards the Cartesian grid by candidate index. Each shard
has independent resume state and evaluation caches. Only the last numbered shard
aggregates after all shards are complete; rerun that cell if it finished early.
The weighted candidate and checkpoint are selected solely by mean GCN validation
on search seeds. Disjoint final seeds evaluate initial, unweighted and selected
weighted representatives. The unweighted reference uses its saved validation
curve up to the same budget, including existing extensions when available.

Default cell: Arxiv 0.05%, 90 cells, rank 8, 300 steps. Existing unweighted
resume configurations remain compatible. Weighted runs start from the original
initialization, not from an already optimized low-rank checkpoint. Tests cover
streamed derivatives against dense autograd, normalization, shift invariance,
resume and saved-factor reconstruction. Run tests in Colab; local checks are
limited to AST parsing and diff checks.

`extend_node_weights(previous_run, steps=1000)` restores the original sweep winner's
last Adam checkpoint with its node logits and low-rank factors. It does not rerun the
grid. Outputs go to that candidate's `extended` directory. Original and new validation
checkpoints compete for selection, while final student seeds remain disjoint. Existing
unweighted extension results are reused up to the same budget; the returned baseline
curve makes its available budget explicit. Repeating the call resumes saved state.
