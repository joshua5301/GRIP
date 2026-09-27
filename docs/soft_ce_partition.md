# Soft assignment with an implicit CE student

This experiment replaces the inner ridge learner with a linear softmax learner.
Only assignment logits L are optimized. P = row_softmax(L), pi_j = sum_i P_ij/N,
c_j = sum_i P_ij z_i/(N pi_j), and s_j = sum_i P_ij q_i/(N pi_j).
The features z remain centered, RMS-scaled S²X. The saved teacher probabilities q,
teacher gamma and label temperature T remain unchanged. Initial assignments use
the previous hard partition with the same uniform mixing, not a learned partition.

With x_j = [c_j,1] and theta shaped classes x (features+1), the inner problem is

L_in(theta,P) = sum_j pi_j CE(s_j, softmax(x_j theta^T)) + lambda ||theta||_F²/2.

The outer problem is

J(P) = mean_i CE(q_i, softmax([z_i,1] theta*(P)^T)).

Positive lambda regularizes weights and bias, giving a unique inner minimizer.
The outer objective has no additional regularization. Both objectives use ordinary
softmax logits: the old ridge score temperature tau is not applied. No validation
or test labels enter optimization or checkpoint selection. The default lambda
3e-5 comes from the previous validation-based linear CE diagnostic, not a new sweep.

## Implicit derivative

At the converged inner solution, g = d L_in / d theta = 0. Let H = d g / d theta,
b = d J / d theta. Solve H v = b, then dJ/dP = -(d g/dP)^T v. The implementation
first computes this vector-Jacobian product for the cell moments, differentiating
their masses, centers and labels together. The existing chunked AssignmentMoments
backward propagates it through the softmax assignment logits.

For p_j = softmax(x_j theta^T), a direction U has Hessian action

H[U] = sum_j pi_j (sum_k s_jk) (diag(p_j)-p_j p_j^T) (U x_j) x_j^T + lambda U.

The label sum is retained to match floating-point soft targets exactly. Hessian
actions and their diagonal preconditioner are analytic. No full Hessian, inverse,
or unrolled inner-training graph is stored. Preconditioned conjugate gradients
checks the actual residual b-Hv before declaring convergence. No damping term is
added to the mathematical Hessian; the positive inner regularization supplies it.

Inner learning uses float64 L-BFGS, warm-started from the preceding outer step.
If needed, a bounded Newton/CG line-search polish improves stationarity. A failed
inner stationarity check or implicit CG residual check stops before updating L,
saves diagnostics, and raises an error. Small residuals are diagnostics rather
than an unconditional accuracy guarantee: inverse curvature can amplify errors.

Outer Adam uses float32 logits, normalized by the initial teacher CE as in the
ridge experiment. Every evaluated step records inner and CG residuals, objective,
mass statistics and wall time. The best converged inner solution by teacher CE is
retained, including step zero. The last evaluation requires no implicit solve;
its CG fields are marked unused with status `evaluated` and NaN residuals.
A fixed step budget is not an outer convergence certificate.

## Comparison and artifacts

`run_soft_ce` consumes the prior Arxiv ridge-inner / CE-outer directory. It checks
graph/split, teacher and assignment signatures, fingerprints representative files,
and confirms that the initial soft moments agree. The five conditions are:

- hard_baseline: original variance–moment hard partition, not GRIP.
- soft_initial: shared mixed initialization.
- ridge_optimized: previous ridge-excess optimized partition.
- ce_optimized: previous ridge-inner / CE-outer partition.
- ce_bilevel: new CE-inner / CE-outer partition.

Every condition is evaluated by a fresh, converged CE linear fit from zero using
the same fixed lambda, including bias regularization. This diagnostic uses the
serialized float32 features/labels actually supplied to GCN, transformed into z
coordinates; optimization itself uses float64 cell moments. The small rounding
difference can make final reported CE differ from best_J. Test is evaluated only
after the assignment checkpoint is chosen.

Existing GCN evaluations are reused. Only ce_bilevel trains new two-layer GCNs
with mass CE, synthetic A=I and the original settings/seeds. Representative inputs
remain weighted S²X means, not reconstructed raw X. A matched CE linear inner
learner still does not guarantee transfer to GCN.

Outputs include config.json, optimization.csv, optimized.pt, representative/head
files, summary.csv and students.csv. Optional best_assignment_logits.pt stores the
selected P via its logits. At N x 909, dense logits and Adam state still use O(Nm)
memory; chunking avoids an additional full probability graph. Completed results
are cached. Interrupted optimization restarts; optimizer state is not checkpointed.

Tests check the analytic Hessian/diagonal and CG against a dense autograd Hessian,
the outer gradient against autograd, assignment hypergradients against finite
differences with re-solved CE heads, saved checkpoint fidelity, and rejection of
unconverged inner solutions. Run numerical tests on Colab, not locally.

## Checkpoint trajectory

Pass `checkpoint_steps=[0,100,200,300,500,750,1000]` with `steps=1000` to
`run_soft_ce` to evaluate one optimization trajectory at several iterations.
Step zero and the last step are automatically included for a nonempty schedule.
Snapshots contain the current moments and converged head at the requested step,
not the lowest-loss iterate seen so far. They are saved immediately under
`checkpoints/step_XXXXXX.pt` and in optimized.pt. Dense assignment matrices are
not saved per checkpoint. Snapshotting does not alter updates or learning rates.

All checkpoint evaluations occur after optimization, using the same GCN seeds
and settings. Fresh linear CE heads are fitted from zero as in the base protocol.
The four historical controls retain their saved GCN results. New rows are named
ce_step_XXXXXX and expose checkpoint_step. The checkpoint with largest mean GCN
validation accuracy is written to selected_checkpoint.csv; ties prefer the earlier
step. Neither linear validation nor any test statistic selects the checkpoint.
Test curves are retrospective diagnostics, not stopping criteria. Seed standard
deviations reflect student initialization on a fixed condensed set, not independent
condensation runs. Selecting on these validation results is validation tuning.

Older runs do not contain intermediate moments and cannot supply these snapshots;
rerun optimization once with a checkpoint schedule. After completion, cached
optimization and student tables can be reused with the same configuration/revision.
