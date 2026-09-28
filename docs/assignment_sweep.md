# Staged MLP assignment search

`src.assignment_sweep.run_assignment_sweep` starts Cora, Citeseer or Flickr from
their source data, without requiring an Arxiv experiment directory. It fixes a
feature-only MLP encoder (hidden 64, rank 16 by default), free assignment mass,
RMS-normalized S²X for the inner linear CE model, and mass-weighted CE for fresh
two-layer GCN students with identity condensed adjacency. Representatives are
inverse-transformed to propagated features as in the previous experiments.

The teacher uses the existing kernel logistic loss and Nyström feature formula.
Anchors are selected only from the condensation graph. For transductive data,
the condensation graph includes unlabeled nodes; only training-mask labels fit
the teacher. For Flickr the condensation graph is the training subgraph, and
the validation subgraph is propagated separately with its normalized adjacency
and mapped with the training anchors. Gamma is chosen by teacher validation
accuracy, breaking ties by grid order. Test is not used for teacher selection.
This is a teacher-first protocol, not a joint gamma/condensation full sweep.

For each density and T, a fresh original risk partition on S²X and teacher
probabilities provides the initialization. B and partition seed are explicit
fixed settings; no old dataset-specific best partition is silently reused.
The staged grid searches T, inner CE penalty, and assignment learning rate.
Stage one evaluates saved steps through 300 with two student seeds. Candidate
ranking uses GCN validation, never outer CE. The best distinct candidates
(default three) proceed to 1000 steps with three seeds. If all promoted candidates
miss another learning rate represented in the grid, one best remaining candidate
from such a rate is added. This reduces but cannot remove early-pruning bias.

Dropout evaluations reuse the same saved moments. Search evaluates validation
only; final selection covers candidate, checkpoint and dropout using refine
validation scores. Only the selected configuration gets ten independent student
seeds and test evaluation. Final validation/test are reported without using
final-seed validation to select a different configuration. Search, promotions,
selection, per-seed results and selected representatives are saved separately.
Independent dataset sessions use separate output roots. Config fingerprints
include data digests, settings and revision. Cached student evaluations and
completed trials are reused on rerun under the same fingerprint.

`optimize_ce_assignment(..., save_resume=True)` saves `resume.pt` at requested
checkpoints and the endpoint via a temporary file replacement. It stores current
parameters, Adam state, warm-start head, dual, original loss scale, best state,
history and snapshots. `resume_state` checks the input digest and solver settings,
then resumes from that evaluated step; the boundary evaluation is repeated before
its next update. It preserves optimizer moments rather than restarting Adam.
Steps and checkpoint requests may change on extension. Floating-point inner
solves can still produce small trajectory differences. This is separate from
the legacy representative-only recovery files, which remain non-resumable.

The staged search reduces assignment updates from 18*3000=54000 to
18*300+3*700=7500 for the example grid, or 8200 if a fourth candidate is retained.
This is not a wall-clock guarantee: head solves and repeated student evaluation
can dominate, particularly with high-dimensional Cora/Citeseer features.

## High-dimensional CE Hessian solves

When augmented feature width exceeds twice the number of cells and the reduced
system has at most 2048 coordinates, the CE solver uses an exact row-space solve
instead of PCG. Otherwise it first tries PCG, then uses the reduced solve as a
fallback when small enough (or retries PCG with four times the iteration budget).
For X=U S V^T, the data Hessian acts only along V in feature coordinates. Its
orthogonal complement is exactly lambda*I, so that component of the right-hand
side is solved by division by lambda. The projected class-by-feature Hessian is
solved directly. No singular directions are truncated and no extra damping or
tolerance relaxation is introduced. The original full-coordinate residual must
still satisfy the requested tolerance or optimization stops. The same solver
supports inner-head Newton polishing. Logs include hessian_solver and the reduced
dimension; cg_residual denotes the verified residual for either solver.


## Joint full grid, maximum 200 assignment updates

Set `full_grid=True`, `assignment_steps=200` and
`checkpoint_steps=(25, 50, 100, 150, 200)` in `run_assignment_sweep`.
Every gamma x T x penalty x assignment_lr candidate runs to the full budget;
there is no promotion or early elimination. Teacher validation remains diagnostic:
gamma is selected jointly using the final student's validation accuracy.
Student training epochs are controlled separately by `epochs`.

Step zero is evaluated once per gamma/T/dropout and competes as a baseline.
Its penalty and assignment learning rate are not identifiable; selected output
uses null for these fields. `baselines.csv` stores these unique baselines.
`boundaries.csv` profiles the best validation per parameter value, excluding
step zero from penalty and assignment learning rate profiles. An edge winner
indicates a reason to expand the grid, not proof of a global optimum.

`full_grid.csv` stores all evaluated checkpoints. Search uses `search_seeds`;
only the globally selected configuration is evaluated with `final_seeds` on test.
Use disjoint seeds. Final evaluation does not choose another configuration.
Resume by rerunning identical settings and code revision against the same Drive
output directory. Each candidate saves optimizer state at checkpoints; student
scores are cached separately. An interruption may repeat work after the latest
saved checkpoint. The staged runner remains the default.


## Faster inner/adjoint tracking

Pass `solver=dict(solver_mode="tracking", tracking_inner_steps=2,
tracking_cg_steps=8, tracking_refresh=20)` to use warm, bounded solver updates
between exact corrections. The full grid, checkpoint selection, gamma search,
GCN seeds and mass-weighted CE are unchanged. See [tracking details](ce_tracking.md)
for residual diagnostics, numerical limits, timing and resume semantics.
