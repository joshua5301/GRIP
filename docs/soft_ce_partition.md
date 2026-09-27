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
marginal tolerance within the balancing budget stops the run. Equality is numerical,
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

The cached backend now continues its existing scaling iterations up to
4*balance_steps when needed, rather than failing at the initial budget. It seeks
0.95*balance_tol before materializing P to leave room for summation roundoff and
still checks the actual marginals against balance_tol. It does not accept a failed
constraint or restart the scaling sequence. Logged balance_iterations includes
the extension; the limit remains bounded.

`evaluate_saved_ce_checkpoints(failed_run, ...)` evaluates snapshots already saved
by an interrupted run without optimizing assignments again. It checks experiment
provenance, includes only files actually present, and writes a separate evaluation
directory with evaluation_only and recovered_steps in its config. It does not
fabricate a final checkpoint or mark the failed optimization complete. In
particular, saved moments cannot resume Adam: no intermediate assignment logits
or optimizer state were stored by the previous implementation.

## Low-rank residual assignments

Set `assignment_rank=32` in `run_soft_ce`; `None` retains dense optimization.
The logits are L=L0+UV^T/sqrt(r), where L0 encodes the same smoothed hard
assignment as before. Only U (N x r) and V (m x r) are optimized. U starts at
zero and V at standard normal using a local generator controlled by `factor_seed`.
Thus initial logits and representatives are preserved relative to the same mass
mode. Both factors cannot start at zero because that would make both gradients
zero. Uniform mode still balances L0 and changes the free-mode initialization.

This limits the rank of the learned logit correction, not of P or total L.
Softmax and balancing can produce full-rank P. The factorization is a restriction
and implicit optimization bias, not a guarantee of better GCN accuracy. The inner
CE, outer CE, teacher, feature normalization, reconstruction and GCN evaluation
remain unchanged. Factor Adam steps are not equivalent to dense-logit Adam steps;
the same learning rate is only an initial controlled comparison.

Free mode evaluates moments and their exact first-order factor derivatives in
blocks, recomputing probabilities in backward. It never stores full N x m logits,
probabilities, logit gradients or Adam states. Assignment working memory scales
with (N+m)r plus chunk_size*m and the material/moments. Arithmetic still includes
all node-cell pairs and additional factor products; speedup is not guaranteed.
Uniform mode composes block-generated dense logits with the existing balancing
backends. Factor parameters and Adam states shrink, but dense temporary logits,
their gradients, and (for cached balancing) float64 P remain O(Nm).

The best outer-CE assignment is saved as `best_assignment_factors.pt` containing
u, v, assignment, mixing and rank. Reconstruct blocks with `logit_block`, then
row-softmax in float64. Uniform runs additionally require the saved column dual.
Checkpoint moments, validation-based selection and interrupted-run evaluation
work as before; factor files represent the best outer CE, not necessarily the
validation-selected checkpoint. These files do not contain resumable Adam state.
Rank and factor seed enter the experiment fingerprint. Tests cover preservation
of initial moments, factor gradients against dense autograd and finite differences,
balanced marginals, and saved best-factor reconstruction for both mass modes.

## Feature-conditioned assignment

With `assignment_rank=16`, set `assignment_input='features'` or
`'features_labels'` to replace independent node embeddings with U=HW.
H is the fixed RMS-normalized propagated feature z, or the concatenation [z,q].
q is the same fixed teacher probability used in the existing representative
labels and outer CE, at the unchanged teacher temperature T. No ground-truth
validation/test labels enter H. There is no additional label scaling, input
normalization, bias, activation, or encoder regularizer.

W starts at zero; cluster embeddings V use the same seeded standard normal
draw for both input modes. All input modes therefore start from the same L0
(or its balanced projection in uniform mode). W and V alone are optimized.
The first update moves W; V initially has zero gradient. The low-rank moments
backward supplies dU, and ordinary autograd computes dW=H^T dU. Free mode still
streams node-cell pairs; H and the derived N x r embeddings occupy memory,
but there are no trainable N x r parameters or their Adam states.

The learned correction is H W V^T/sqrt(r), a low-rank linear score in H.
It is not a nonlinear encoder, nor does it change the feature space used by
the CE student or the representative moments. Features and labels remain
weighted means in the original z,q spaces and are reconstructed as before.
At Arxiv d=128, C=40, m=909 and r=16, the two input modes have 16,592 and
17,232 trainable assignment parameters. This is a stronger restriction than
independent node embeddings; equal Adam learning rates do not imply equal
changes to assignment probabilities.

`best_assignment_encoder.pt` saves W under `weight`, V, input mode, assignment,
mixing, rank, seed and best outer-CE step. Reconstruct H from the original z,q
using `assignment_inputs`, then use `logit_block(H @ weight, v, assignment,
mixing)` and float64 row softmax (plus the saved dual in uniform mode).
The run config retains source digests and transform provenance. Input mode is
part of the fingerprint, and old dense/independent-node options remain valid.

### MLP assignment encoder

Set `assignment_encoder='mlp', encoder_hidden=64` with a feature input mode.
The map is ReLU(H W1+b1) W2+b2, with output width assignment_rank. W1 uses
a seeded uniform initialization with bound sqrt(6/input_dim), b1 is zero,
and W2,b2 are zero. Cluster embeddings V match the linear encoder's seeded
draw exactly; initialization uses local generators and does not advance the
global RNG. Both maps start at identical assignment logits. At the first step
only the output layer can move; the hidden layer and V can learn subsequently.

The MLP uses no dropout, extra normalization or weight decay. Original-node
hidden activations are retained for backward; assignment moments remain streamed
in free mode. Inner/outer CE and representative construction are unchanged.
Hidden width and encoder type enter MLP experiment fingerprints. Saved encoder
files contain `encoder_parameters` in W1,b1,W2,b2 order. `saved_encoder_nodes`
reconstructs U from z,q for both legacy linear and MLP files; uniform assignments
still need the separate column dual. Arxiv features+labels with hidden=64,
rank=16 has 26,400 trainable parameters including V. This increases assignment
expressivity but does not make the inner student nonlinear or guarantee better
GCN transfer. Tests check zero-output initialization, gradient flow to earlier
layers after an update, dense-autograd agreement and saved-state reconstruction.
