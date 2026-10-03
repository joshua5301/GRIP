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

All five datasets are supported. Flickr and Reddit preserve their separate
training, validation and test graphs during GCN evaluation; clustering uses
training-graph features and teacher labels only. Cora and Citeseer use dense
node-to-cell distances. Arxiv, Flickr and Reddit use an exact factorized representation of row-centered
negative squared distances: [2z_i,1] dot [c_j-mean(c), -(||c_j||²-mean||c||²)].
The median positive nearest-center gap is measured in node blocks. QR of both
factors followed by SVD of the small product yields the distance-logit SVD
without forming N-by-m distances. The learned assignment moments and their
backward pass recompute logits in blocks; no dense probability cache is used.
This saves memory at the cost of recomputing the fixed distance blocks.

## Sequential first-density sweep

`src.distance_finetune_all.run_all_distance_finetune` runs Cora 0.013 (35),
Citeseer 0.009 (30), Flickr 0.001 (44), Reddit 0.0005 (77), and Arxiv 0.0005
(90), sequentially, with both methods. These are the first benchmark densities;
Arxiv 909 belongs to ratio 0.005 and is not reused for the 90-cell run.

Matching supplied Var-Part sources are reused. Missing or different-density
sources trigger a normalized-variance stage-one sweep: gamma is selected by
teacher validation accuracy, and T/alpha by uniform-GCN validation. Stage two
freezes these settings and sweeps rank, tau and inner penalty. Ranks larger
than the representative budget are omitted. Inner CE is mass weighted; final
and search GCN CE is uniform. Condensation seed is zero. New stage-one runs use
five search seeds and ten independent final student seeds; reused sources keep
their original seed lists and student settings.

All source runs, resumes, temporary files and summaries live below output_dir;
raw data remain below data_dir. summary.csv and status.json update after each
method. A failed dataset or method is recorded and the others continue. An
individual numerical solver failure is recorded in failures.csv and the next
candidate runs. status.json reports failed candidate counts; a completed run
with nonzero failures is a partial grid, not a complete successful grid. An
identical rerun reuses cached evaluations and optimization checkpoints, so it
does not require the notebook's in-memory state. Solver or source changes can
create a new run fingerprint. Completed method paths remain in status.json.

Set `inner_loss_weightings=["mass", "uniform"]` to sweep both inner objectives
with the same stage-one partition, seed lists and hyperparameter grid. GCN
search and final evaluation remain uniform CE. Separate `inner_mass` and
`inner_uniform` folders prevent cache mixing. Each weighting reports its own
validation-selected configuration and checkpoint. selected_by_validation.csv
then chooses the inner weighting per dataset/method by search validation,
never by test. This doubles the stage-two candidate count, not the stage-one
teacher/partition sweep.
