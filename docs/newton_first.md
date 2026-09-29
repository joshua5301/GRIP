# Newton-first inner CE solver

`solve_inner_newton_first` starts from the previous head and first checks its
stationarity. Up to eight damped Newton corrections are attempted, each with an
Armijo line search. Newton direction PCG uses a relative tolerance between 1e-4
and 0.1 based on the current maximum gradient. This intermediate tolerance does
not replace the final head gradient criterion. If corrections do not satisfy the
original `grad_tol`, the existing L-BFGS plus Newton-polish solver runs from the
last accepted head. With no initial head it uses the existing solver immediately.

`optimize_ce_assignment(inner_method='newton_first')` enables it explicitly.
The default remains L-BFGS until the timing comparison is assessed. New production
runs still warm-start implicit Hessian solves by default. Existing resume settings
are not silently converted to a different inner method.

`benchmark_newton_first` compares both solvers on the exact same condensed data
and initial head at every step. Solve order alternates, repetitions are timed with
CUDA synchronization, and the first five updates are excluded. Reference L-BFGS
heads drive the common assignment trajectory. The implicit solver uses warm starts
for that trajectory: the benchmark changes only its in-memory copied source resume
configuration and never writes to the original experiment. It does not evaluate GCN
students or alter their cached results.

Report time including Newton fallback, final gradient convergence, objective gap,
parameter difference and predictive-probability difference on condensed inputs.
The reported total-step gain is an estimate, not a separate end-to-end Newton-first
training measurement. A faster solve with failed convergence is not an improvement.
Completed timing windows are cached; interrupted windows restart from the source.
Numerical tests are supplied for Colab, with only syntax/diff checks run locally.

`compare_newton_performance` reruns the selected low-rank configuration from step
zero with each inner solver independently. Both use implicit warm starts, identical
cached inputs, initial factors, teacher labels and tolerances. Only the inner method
differs. Original artifacts remain untouched. Each trajectory resumes from its own
checkpoints and GCN evaluations are cached by seed.

Report both the fixed 1000-step result and the checkpoint selected by three GCN
validation seeds, then evaluate with ten separate paired seeds. GCN training uses
two layers, identity synthetic adjacency and mass CE, matching the source. Test
scores never choose checkpoints. The output includes per-seed differences and
validation trajectories. Optimization timings exclude downstream GCN evaluation;
the summed update times exclude checkpoint I/O, while condensation_seconds includes
optimizer bookkeeping and most I/O. These sequential timings are not a repeated
hardware benchmark. One condensation seed does not establish seed-wide equivalence.
