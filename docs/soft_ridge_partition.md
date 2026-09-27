# Direct soft-assignment ridge condensation

The only optimized variable is a dense N x m logit matrix L. P = row_softmax(L)
assigns each original node fractionally to cells. For cell j, n_j = sum_i P_ij,
c_j = sum_i P_ij z_i / n_j, s_j = sum_i P_ij q_i / n_j and pi_j = n_j/N.
Features, labels and masses are coupled through P; none is independently trained.
No entropy, uniform-mass, sparsity or balance penalty is added.

## Exact ridge learning objective

z is the same centered, RMS-scaled S²X used by the earlier Arxiv experiment.
An appended constant coordinate supplies a bias, regularized along with weights.
For X = [Z, 1], define G = X^T X/N and H = X^T Q/N. The full reference is
W* = solve(G + lambda I, H). A condensed ridge head is fitted from cell means
and fractional masses using the same positive lambda.

The optimized objective is exactly

J(P) = F(W_P) - F(W*)
     = 1/2 tr((W_P-W*)^T (G+lambda I) (W_P-W*)),

where F(W) = ||XW-Q||_F²/(2N) + lambda ||W||_F²/2. The difference formulation
avoids cancellation of two nearly equal loss values. Chosen lambda minimizes no
partition quantity: it is selected before assignment training by full-data ridge
validation accuracy, breaking ties with ground-truth validation MSE. Reference
test is evaluated only after this selection. Ridge scores are unconstrained
regression outputs; their argmax is used for classification, not a softmax CE fit.

## Implementation and memory

Initialize P = (1-mixing) P_hard + mixing/m. The default mixing is 0.05.
Logits and Adam state are float32. Softmax, aggregation, ridge solves and objective
evaluation use float64; TF32 is disabled. Material T=[1,Z,Q] is fixed. A custom
autograd operation computes P^T T/N in node chunks. Its backward recomputes each
chunk's probabilities and applies the exact softmax Jacobian-vector product.
It supplies gradients only for L, intentionally not for fixed material T, and
implements first-order differentiation for this optimizer. No dense N x m
probability matrix or all per-chunk backward graphs are retained. L and Adam's
state still scale as O(Nm); chunking does not make assignments sparse.

Adam minimizes J divided by its initial value (floored at 1e-12 for numerical
scaling), which has the same minimizers. Small eps=1e-12 prevents Adam's default
epsilon from dominating per-node gradients. Current objective can increase:
the final representative set is the lowest exact outer objective encountered,
including the soft initial state, never selected by validation or test. A fixed
step budget is not a convergence certificate.

## Controls and saved results

`run_soft_ridge` reuses the saved Arxiv teacher logits and baseline assignment,
verifies graph/split signatures and baseline label means, then compares:

- `hard_baseline`: original hard partition, evaluated with the new ridge head.
- `soft_initial`: after mixing, before learning assignments.
- `soft_optimized`: best soft assignment by full-data ridge objective.

All conditions report actual ridge excess objective and classification accuracy.
As a separate transfer experiment, the same representatives train fresh two-layer
GCNs with synthetic A=I, mass CE and paired seeds. This does not imply that ridge
optimization guarantees GCN improvement. Representative raw inputs are the inverse
affine transform of c_j, hence weighted means of S²X. There is no neural preimage
optimization or argmax hardening. GCN hyperparameters and schedule remain the
source experiment's settings. The final GCN has no ridge penalty substituted for
its original weight decay.

The root stores reference grid and selected reference, `optimization.csv`,
`optimized.pt`, three representative files, `summary.csv`, and `students.csv`.
Optional `best_assignment_logits.pt` reconstructs the selected soft P by row
softmax and can occupy roughly 0.6 GB at Arxiv/909 in float32. Logs are written
during training; completed optimization/student tables are reused. Interrupted
Adam training restarts because optimizer state is not checkpointed. Progress
tables alone are not completion markers. `effective_cells` = 1/sum_j pi_j²
diagnoses mass imbalance; all cells retain soft positive mass absent underflow.

Local numerical training/tests are not part of the workflow. Colab tests check
chunked gradients against dense autograd and finite differences, mixing and mean
preservation, exact objective identity, and selected assignment checkpoint fidelity.

## Cross-entropy outer objective

`run_soft_ridge_ce(previous_run, ...)` retains the previous ridge penalty, teacher
probabilities, hard partition, mixing, GCN settings and evaluation seeds. It starts
from the original mixed hard assignment, not the ridge-optimized assignment.
The inner solution remains weighted ridge regression on the coupled cell means:

W(P) = solve(C_aug^T diag(pi) C_aug + lambda I, C_aug^T diag(pi) S).

Only the outer objective changes to

J_CE(P) = -(1/N) sum_i sum_k q_ik log softmax([z_i,1] W(P) / tau)_k.

There is no extra outer weight penalty. Differentiation passes through the ridge
solve and cell means to assignment logits. Unlike ridge excess, CE requires logits
on all original nodes each step; it cannot be computed from G and H alone.
Its minimum is not assumed to occur at the full-data ridge solution.

Ridge outputs are regression scores. The separate positive score temperature tau
is selected once by ground-truth validation CE of the full-data ridge reference,
then frozen for optimization and every comparison. It does not change teacher
label temperature T or argmax predictions of a fixed head. Use
`score_temperatures=[1.]` to disable calibration. No validation or test labels
enter the assignment objective; validation does inform the fixed calibration.
The checkpoint is selected solely by teacher CE on all nodes. This objective
still targets teacher predictions rather than ground-truth accuracy.

The wrapper verifies comparison provenance and reuses previous GCN evaluations
for `hard_baseline`, `soft_initial` and `ridge_optimized`. Only `ce_optimized`
needs new GCN fits. All four ridge heads are reevaluated with the same tau and
report both teacher CE and ridge excess. Previous ridge J and current CE J have
different meanings and must not be compared numerically. `temperature_grid.csv`
records calibration; `optimization.csv` now tracks CE when outer_loss is ce.
Tests additionally check CE gradients against dense autograd and finite
differences, and ensure saved checkpoints minimize the chosen outer objective.
