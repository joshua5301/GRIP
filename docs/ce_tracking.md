# Tracking CE bilevel assignments

`solver_mode="tracking"` preserves the CE inner objective, regularization,
CE outer objective, assignment family, moment reconstruction and GCN evaluation.
It changes the numerical optimization trajectory. It is an inexact implicit
solver with periodic correction, not an implementation of the SOBA convergence
theorem and not a claim of identical gradients or final accuracy.

For assignment parameters psi and head W, the target remains

    min_psi F(W*(psi)), W*(psi) = argmin_W L(W, psi)

The adjoint solves H v = grad_W F, where H = Hessian_W L. The assignment
hypergradient is - (d grad_W L / d psi)^T v. Positive penalty applies to the
whole head, including bias. The original exact solver remains the default.

## Updates

Start with a converged inner head and adjoint. Between refreshes:

- Warm-start the head from the previous iteration. Take at most two Newton
  updates, each using eight diagonal-preconditioned CG iterations and an Armijo
  line search. No SVD or dense Hessian factorization is used in this path.
- Warm-start the adjoint from the previous iteration. Take at most eight PCG
  iterations for the new Hessian and outer gradient. Restart CG directions for
  the changed system; carry the solution, not old conjugate directions.
- Use this approximate head and adjoint to update the existing assignment MLP.

Defaults are `tracking_inner_steps=2`, `tracking_cg_steps=8`,
`tracking_refresh=20`. This bounds the ordinary step's solver work; it does not
certify its hypergradient accuracy. Smaller penalties can require more tracking
work or more frequent correction. A finite but unconverged residual is recorded,
not treated as a converged solve. No unrolling graph is retained across updates.

Every 20 updates, at every requested checkpoint, and at initialization and the
final update, use the original inner and implicit solver tolerances. Failed
tracking line searches or nonfinite head states invoke exact head correction;
nonfinite adjoints invoke the exact Hessian solver. A failed exact correction
still stops the run. The adjoint is unnecessary at the last step.

`J_exact` means the inner stationarity tolerance passed. Only these CE values can
update `best_J` or its saved partition. All evaluation snapshots have a converged
inner head. Approximate CE values remain in the history for diagnosis, but must
not be compared as exact bilevel risks. GCN selection still uses validation;
test is evaluated only after the globally winning configuration is fixed.

## Logging and resumption

`optimization.csv` includes `exact_refresh`, `head_fallback`,
`implicit_fallback`, `inner_grad_max`, `cg_relative_residual`, `J_exact`, and
per-phase timing columns. `head_correction_relative` and
`implicit_correction_relative` measure the change from the preceding carried
state at the current assignment; these are not formal hypergradient error bounds.

Resume files include the incoming head and adjoint as well as assignment
parameters and Adam state. Replaying a saved checkpoint therefore does not apply
an extra tracking update. Resume with the same data, settings, checkpoint schedule,
and code revision. Solver modes use separate sweep cache fingerprints; exact-run
scores are not silently reused as tracking-run results.

## Use and comparison

Pass these options through `run_assignment_sweep(solver=...)`, or directly to
`optimize_ce_assignment` / `run_soft_ce`:

    solver_mode="tracking"
    tracking_inner_steps=2
    tracking_cg_steps=8
    tracking_refresh=20

The complete grid and 200-update budget remain unchanged. Student training still
costs the same: tracking accelerates condensation, not repeated GCN evaluation.
For a speed-quality comparison, use matched settings and seeds with `exact` and
`tracking`, report condensation and student-evaluation time separately, and
compare validation versus elapsed time as well as the final score. Fixed update
counts alone are not a speed-quality comparison.

Run `tests/test_ce_tracking.py` and the existing CE/assignment tests in Colab.
Local verification is limited to syntax and diff checks.
