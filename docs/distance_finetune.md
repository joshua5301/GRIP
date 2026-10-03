# Distance-initialized bilevel finetuning

Freeze gamma, teacher T and alpha selected by stage-one normalized variance
Lloyd clustering. Construct node-to-final-center squared distances in the same
normalized joint feature/label space. Divide by the median positive nearest vs
second-nearest distance gap, then remove row means (softmax invariant).

Compare fixed D plus low-rank correction, (-D/scale + UV^T/sqrt(r))/tau,
against pure UV initialized by truncated SVD of -D/(scale*tau). For the latter,
balanced SVD factors include r^(1/4) in both factors to match the existing
1/sqrt(r) logit convention. A zero base is used only to reuse the chunked solver;
no distances remain in its learned assignment formula.

Both optimize original-node teacher CE through the exact regularized linear CE
inner student. Inner CE defaults to cell-mass weighting; GCN evaluation uses
uniform CE. Labels and features remain assignment-weighted means. Rows of P
sum to one; columns have free mass. Newton-first and implicit Hessian warm
starts reuse existing solvers. Checkpoint and hyperparameter selection use GCN
validation only; student search and final seeds are inherited from stage one.

Reports distinguish hard stage one, soft step zero, selected finetuning, and the
full-distance soft reference at the selected SVD temperature. Step zero is an
eligible checkpoint. Citeseer stores only the small N-by-m base, not N-by-N
distances; moments and their gradients use node chunks. Resume states are saved
at checkpoints, and completed optimization/student artifacts are reused.

The `methods` argument can restrict the sweep to fixed_D only. Initial GCN
search evaluations are shared across ranks and penalties at fixed tau for
fixed_D, and across penalties at fixed rank/tau for svd_UV. No SVD is computed
in fixed_D-only runs.
