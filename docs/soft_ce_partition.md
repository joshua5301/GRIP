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

## Uniform cell mass

Set `mass_mode='uniform'` to constrain sum_j P_ij=1 and sum_i P_ij=N/m.
The objective and CE head are unchanged. BalancedMoments solves
P_ij = softmax_j(L_ij + v_j), adjusting column potentials v until the largest
relative column residual is at most balance_tol (default 1e-8). Row sums follow
from softmax. This is the KL projection of the row-softmax assignment onto the
balanced transport constraints; it does not add a tunable entropy penalty to J.
Warm-started potentials accelerate subsequent evaluations. Failure to meet the
marginal tolerance within balance_steps stops the run. Equality is numerical,
not exact arithmetic. No stop-gradient approximation is used for balancing.

For an upstream moment gradient, let G be the unnormalized gradient for P,
a_i=sum_j P_ij G_ij and pi=P^T 1/N. The column derivative solves

(diag(pi)-P^T P/N) b = P-weighted column sums of (G-a)/N.

The all-ones null direction is removed by a gauge term 11^T/m, not damping.
The logit gradient is P_ij (G_ij-a_i-b_j+sum_k P_ik b_k)/N.
The original `balance_backend='chunked'` aggregates in chunks and builds a dense
m x m Gram matrix, costing O(N m²) work. Balancing also adds repeated O(Nm)
passes. This backend remains available as a numerical reference and lower-memory
fallback.

Uniform runs start from the same assignment logits as free runs, then balance
them. Their step-zero representatives therefore differ; retain step zero to
separate balancing at initialization from subsequent learning. Checkpoints and
the best representative set retain equal masses within tolerance. Existing hard,
soft-initial and ridge controls remain explicitly labeled mass_mode='free'.
Under equal masses, mass-weighted CE equals uniform CE, while retaining the
same evaluation code. The output reports mass_tv and mass_relative_residual.

To reconstruct a saved best uniform P, load both best_assignment_logits.pt and
best_assignment_column_dual.pt, then take row_softmax(logits.double()+dual).
The logits alone do not describe the balanced assignment. Numerical tests cover
marginals, global moments, finite-difference derivatives, gauge invariance,
nonconvergence rejection, and uniform bilevel checkpoint reconstruction.

## Cached balancing and phase timings

The default `balance_backend='cached'` builds exp(L-row_max(L)) once, performs
Sinkhorn scaling with matrix-vector products, and retains the resulting float64
P through backward. It replaces repeated chunk softmax/exponential evaluations.
Warm-started column potentials and the marginal tolerance remain unchanged.
Underflow or marginal failure raises an error; the chunked backend is available
for extreme logits requiring log-domain evaluation.

Backward solves the same gauge-fixed balancing derivative with preconditioned CG.
Products use pi*b - P^T(P*b)/N + mean(b), without constructing P^T P. This costs
O(k N m) for k CG iterations rather than O(N m²), plus moment-gradient products.
`balance_cg_steps=512` and `balance_cg_rtol=1e-7` control this solve independently
of the CE-head Hessian solve. Actual residuals must meet the same default relative
threshold used by the original dense solve (and absolute floor 1e-12). There is
no new penalty, relaxed mass constraint, or precision reduction. Tolerance-based
iterative solves can still produce small numerical trajectory differences.

The P cache takes 8Nm bytes, about 1.3 GB for this Arxiv/909 experiment. Logits,
Adam states and temporary arrays are additional. Overall peak memory must be
measured on Colab; no runtime speedup has been measured locally. Tests compare
both backends and their gradients, and check finite differences and the CG solve.

`outer_chunk_size=65536` batches original-node CE evaluation separately from the
assignment chunk size, reusing an augmented feature matrix. `log_every=10` reduces
Drive writes while retaining every history row in memory and the completed log.
Failures and the last step always write diagnostics. GPU-synchronized phase timers
record assignment_seconds, inner_seconds, outer_seconds, implicit_seconds and
backward_seconds (including the assignment optimizer update). Snapshot/host IO
overhead remains in total elapsed seconds and is not assigned to these five phases.
