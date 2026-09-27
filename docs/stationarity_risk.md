# Stationarity-preserving partition refinement

This Cora diagnostic starts from the saved S²X baseline partition of
`run_learning_audit`. Features, teacher probabilities, initial partition, student
hyperparameters and final student seeds are shared between conditions. It is
not a full hyperparameter sweep or a comparison of independently tuned methods.

## Objective

Let z be centered, RMS-scaled S²X, and let x = [z; 1]. A head Theta includes
both weights and bias. Both are regularized:

F(Theta) = mean_i CE(softmax(Theta x_i), q_i) + lambda ||Theta||_F² / 2.

A partition uses arithmetic feature/label means c_j and s_j, with mass n_j/N.
The synthetic gradient at a frozen full-data reference Theta_ref is

G_S = sum_j (n_j/N) (softmax(Theta_ref [c_j;1]) - s_j) [c_j;1]^T
      + lambda Theta_ref.

Refinement minimizes J = ||G_S||_F². It does not subtract the numerical
full-data reference gradient: this is the actual synthetic stationarity residual.
The reference is fit to all nodes' soft teacher probabilities. Lambda is selected
by ground-truth validation CE among converged full-data fits, before refinement.
Test is never used to select lambda or the partition. This is a different linear
head protocol from the previous audit, which did not regularize bias; new baseline
and refined heads are therefore fitted under the same updated protocol.

## What is guaranteed

F_S is lambda-strongly convex in all head parameters. For its exact optimum
Theta_S*, any reference Theta_ref satisfies

||Theta_S* - Theta_ref||_F <= ||G_S||_F / lambda.

For a numerically fitted synthetic head, add
||gradient F_S(Theta_fitted)||_F / lambda. The experiment reports this allowance
and the corresponding bound violation. If the full-data reference is also
approximate, its distance to the exact full-data optimum is at most its gradient
norm divided by lambda. These are linear-head statements, not guarantees for a
newly trained nonlinear GCN or its ground-truth accuracy.

## Solver

Every eligible node is considered for every other cell. Exact finite-move score
changes recompute the two affected cells' means, probabilities and gradient
contributions. Candidate pairs are chunked to avoid an N x K x C x D tensor.
Proposals within a block use the current partition. A combined proposal is
accepted only after recomputing the full objective; rejected blocks are recursively
split. Cells cannot be emptied. No accepted move in a full sweep is reported as
`no_accepted_move`; hitting the sweep limit is not convergence. There is no global
optimality claim, and monotone J does not imply monotone accuracy.

## Evaluation and artifacts

- Baseline versus the terminal refined partition; no best-test or best-validation
  selection over intermediate partitions.
- Fresh convex linear heads on cell means: reference-gradient residual, parameter
  distance, distance bound, excess full-data regularized teacher objective, and
  ground-truth accuracy. Reference and synthetic convergence diagnostics are saved.
- Fresh two-layer GCNs with A=I on mass-weighted synthetic CE, shared paired seeds,
  original student hyperparameters, and validation-selected checkpoints.
- Representatives are arithmetic S²X means. The linear transform commutes with
  averaging, so realization error should be near floating-point error. This does
  not assert that a nonlinear GCN has the S²X feature map.

Saved files include `config.json`, `reference_grid.csv`, `reference.csv`, head
checkpoints, `stationarity_partition.pt`, `partition_history.csv`, `summary.csv`,
and `students.csv`. Config identity includes graph/split signatures, source config,
initial assignment digest, solver options, seed list and Git revision. Finished
partitions/student tables are reused within that identity. Partial student CSVs
are progress records, not completion markers.

Run `tests/test_stationarity_risk.py` in Colab; no local numerical execution is
required. The tests compare analytic gradients with autograd, finite-move scores
with full recomputation, accepted-objective monotonicity, nonempty cells, the
strong-convexity bound with numerical fit allowance, and singleton handling.

## Arxiv experiment

`src.stationarity_arxiv.run_arxiv_stationarity` builds its own teacher and baseline
without Cora artifacts. The default explicit budget is 909 nodes (the repository's
conventional `arxiv, ratio=0.005` setting). The measured 909 / original-node-count
ratio is saved separately; it is not assumed to equal the rounded benchmark label.

The fixed default teacher gamma, temperature, B and student dropout come from the
earlier Arxiv risk setting at ratio 0.0025, not a newly selected optimum at 909
nodes. Users can supply the full parameter dictionary. Both partitions share it.
The experiment is a mechanism comparison, not a full sweep. Positive linear
penalties are selected by reference validation CE as in the Cora experiment.

For scale, Arxiv defaults to 16 closest other cells in standardized S²X plus
4 random cells per node. Random candidates can repeat. Distances are computed in
node blocks, with no N x N storage. Finite-move gradient changes are still exact.
Gradient contributions are cached, and an accepted block updates only affected
cell statistics. Statistics are rebuilt each sweep to remove accumulation drift.
The unrestricted Cora path remains available with `candidate_k=None`.
Restricted search stopping without a move is `candidate_stalled`, not a certificate
of full single-node local optimality; `partition_converged` remains false there.

Additional full-graph, two-layer GCN controls use (1) true labels on train nodes
only and (2) teacher probabilities on all nodes. They keep the same architecture,
dropout, optimizer and schedule as condensed students, but train on the original
graph. They are not independently tuned best full-graph results. Both select epochs
by validation and use test only for reporting. Full-graph seed count can differ
from the paired condensed comparison and is shown explicitly in the output.
`label_teacher.csv`, `full_students.csv`, and per-reference JSONs record these
baselines. No ground-truth validation/test labels are used in teacher fitting,
partition refinement or student training; they are used only for the stated
validation selection and final test reporting.
