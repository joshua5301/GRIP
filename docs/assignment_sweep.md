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
