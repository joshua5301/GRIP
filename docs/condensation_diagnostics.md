# SGC selection, GCN transfer and overfitting diagnostics

## GCN validation reselection

Use `selection_architecture="GCN", search_seeds=[0, 1, 2]` to select both
hyperparameters and condensation checkpoint by the mean best-validation accuracy
of fresh GCN students. Every supplied candidate and checkpoint, including zero,
is evaluated. Search metrics use train and validation masks only. Final seeds
must be disjoint from search seeds; test labels are evaluated after selection.
The optimizer itself remains CE-linear SGC inner/outer, so this changes selection,
not the condensation objective. This is GCN-tuned evaluation, not blind transfer.

Pass `checkpoint_source` as an earlier method directory containing `config.json`,
`teachers.pt`, `initialization.pt` and all candidate checkpoints to reuse previous
optimization. The runner validates the method, graph/feature digest and every
optimization setting before copying checkpoints into the new experiment. It
does not modify the source. Selection/training settings can change; optimization
settings cannot. The source revision and path are recorded. Partial GCN evaluation
resumes from completed per-seed files. A grid with 72 candidates, 7 checkpoints
and 3 search seeds requires 1512 GCN fits per method, in addition to final
evaluation. `search_gcn.csv` contains selection scores; `search_sgc.csv` remains
available as a diagnostic. For comparison with earlier SGC results, retain the
same grid, checkpoints and GCN training settings.

`src.condensation_diagnostics.run_condensation_diagnostics` compares three methods
under the same teacher grid and CE inner/outer objective. The first experiment is
Cora, ratio 0.052, 140 representatives. Cora, Citeseer and Arxiv are supported.
Arxiv ratio 0.005 uses the repository's configured budget of 909 representatives.

| Method | Initialization | Optimized quantities | Condensed CE |
| --- | --- | --- | --- |
| A | Feature-only k-means++ | Feature-input MLP assignment; induced features, labels and masses | Current cell mass |
| C-mean | Same k-means++ state | Free representative features; fixed initial teacher-average labels and masses | Fixed initial mass |
| C-prior | Random true-class train nodes; counts follow train-label proportions | Free representative features; fixed one-hot labels | Uniform |

All feature materials are S²X. C-prior also samples S²X at the selected train
nodes, not raw X. A and C-mean start from the existing 5% uniform smoothing of
the hard k-means assignments, with zero initial learned assignment correction.
They therefore share exactly the same initial soft means, labels and mass.
Neither runs a GRIP or variance-moment refinement before optimization.
`collect_diagnostics` verifies their step-zero moments after both runs finish.

C-prior allocates integer class counts by largest remainders. Sampling is without
replacement unless a class quota exceeds its available train nodes. For Cora
0.052, the 140 representatives use all 140 true-label training nodes. C-prior
changes initialization, labels and weighting together; A versus C-mean is the
more controlled comparison. A still changes labels and masses with assignment,
so that comparison does not isolate the feature constraint alone.

## Selection and architecture transfer

SGC is the regularized CE linear classifier on RMS-normalized S²X with an affine
head. Its inner fit is solved to the configured gradient tolerance at each saved
checkpoint. The positive L2 penalty applies to the bias as well. This is a
deterministic convex fit, not a stochastic multi-seed average. It is both the
optimization surrogate and the source evaluation model.

Teacher gamma, temperature, inner penalty, assignment/feature learning rate and
condensation checkpoint are selected **only by SGC ground-truth validation
accuracy**. Ties prefer earlier candidate indices and then earlier checkpoints.
The teacher uses only true train labels; its validation scores are logged, but
all supplied gamma values enter the outer grid.

After selection is frozen, every saved checkpoint of that configuration is
evaluated on SGC and fresh two-layer GCNs. GCN sees the same representative
features transformed back from RMS coordinates, the same labels and the same
mass. Its condensed adjacency is I, and evaluation uses the original graph.
GCN dropout/lr/weight decay are fixed in the protocol; GCN validation chooses
only its training epoch, never the condensation setting or checkpoint.

For Arxiv, set `trajectory_seeds=[100, 101, 102]` and
`final_seeds=list(range(100, 110))` to reduce intermediate GCN evaluation cost.
Both step zero and the SGC-selected checkpoint always use all final seeds,
preserving paired before/after evaluation. Other checkpoints use trajectory
seeds; their curve means therefore use fewer runs. By default every checkpoint
uses final seeds, preserving the Cora evaluation procedure. Full-data references
also use final seeds. All candidates still receive the complete condensation
budget and all saved SGC evaluations; no GCN-driven candidate pruning is used.

The Arxiv loader retains its original train/validation/test masks on one
transductive graph and standardizes raw features using train nodes only. The
same CPU-prepared S²X and k-means++ procedure is used in all three sessions.
Assignment moments are already chunked; no dense node-by-cell matrix is saved.
The kernel teacher's N-by-basis feature matrix is still materialized, so use
the intended A100 runtime. Data loading, CPU k-means++ and teacher fitting may
take time before condensation progress bars appear.

If diagonal-preconditioned CG fails, its extended retry warm-starts from the
previous iterate and restarts less often. A failed retry can use a feature-span
direct solve when its dimension is at most 8192. The Arxiv 128-feature, 40-class
affine head has dimension 5160 and fits this cap. The dense system is assembled
in node chunks; this fallback costs quadratic memory in head dimension, not
node count. Acceptance still checks the original Hessian residual at the same
tolerance and penalty. Larger systems still report nonconvergence rather than
silently weakening the criterion.

SGC and GCN test values are recorded after source selection. Do not select a
new checkpoint or grid from these diagnostic test curves. The primary comparison
uses SGC-selected checkpoints; the GCN result has not been tuned for GCN.

An optional full-data reference trains each architecture using the original true
train labels. SGC selects its penalty from the same grid; GCN uses the same fixed
training settings and seeds. `test_drop_from_full` reports full-reference test
accuracy minus condensed test accuracy in percentage points. These references
are not claims about the best achievable full-data accuracy.

## Overfitting measurements

At each condensation checkpoint, `trajectory.csv` records:

- True-label train/validation/test accuracy and CE on the original graph.
- Teacher-target CE, both whole-graph and per split, and condensed training CE.
- Validation CE minus train CE, and train accuracy minus validation accuracy.
- GCN metrics at its best-validation epoch and at the fixed final epoch (`last_*`).

`plot_diagnostics` defaults to fixed-final-epoch GCN results, so changing the
selected GCN epoch does not obscure the condensation-step curves. Pass
`fixed_epoch=False` for the best-validation-epoch view. Each GCN run also saves
its training-epoch history. `plot_student_learning` plots true train/validation
CE and accuracy versus GCN training epoch for each selected condensed dataset.

A stronger overfitting signal is decreasing train CE accompanied by worsening
validation CE/accuracy. A large gap alone is not proof. Teacher errors, label
distribution differences and representation mismatch can also produce it.
Likewise, SGC improvement without GCN improvement is evidence of limited
transfer for this pair, not proof of general architecture independence.

Each method selects its own best source configuration. Differences at those
configurations combine method and hyperparameter effects; do not interpret them
as a pure causal effect of the clustering constraint. For controlled SGC
comparisons, join `search_sgc.csv` on gamma, T, penalty, assignment_lr and step.
Teacher CE values are comparable across methods only when gamma and T match.
True-label CE does not have that target-distribution ambiguity.

The default outer loss uses teacher targets at **all** nodes, retaining the
previous transductive protocol. Validation/test ground-truth labels never enter
the outer gradient, but their graph features and teacher targets do. Set
`outer_scope="train"` for a separate train-only outer-target experiment. Features,
propagation and clustering still use the full graph, so this is not an inductive
holdout experiment. Use a different output directory for that comparison.

## Parallel execution and recovery

Run the same Colab cell in three sessions, changing only `method` to `A`,
`C-mean`, or `C-prior`. A common protocol hash includes the grid, seeds, graph,
CPU-prepared propagated features, optimizer settings and Git revision. Each
method writes into its own subdirectory, including its teacher cache, avoiding
shared concurrent writers. Do not run two sessions for the same method/path.

Re-running the identical protocol restores assignment/feature Adam state, the
head and adjoint at the last saved condensation checkpoint. Finished candidates
and completed GCN seed evaluations are reused. An interrupted GCN seed restarts
that seed. Changing the grid, checkpoint list, settings or Git revision creates
a new protocol directory. To resume across a repository update, use the original
revision recorded in `config.json`.

After all sessions finish, `collect_diagnostics(protocol_dir)` returns all three
summaries, trajectories and initial-state checks. Re-run only the result-reading
section of the cell to update plots. Each method saves its own plots to avoid
parallel overwrites. Standard deviations summarize GCN training seeds only;
they do not include teacher, partition or hyperparameter-selection variation.

The tests cover true-train prior initialization, preservation of custom fixed
targets, restricted outer targets, optimizer resumption, source-only selection,
and best-validation versus final-epoch GCN recording. They are intended to run
in Colab; no local numerical test or training was performed for this change.
