# Paired implicit Hessian timing

`benchmark_low_rank_hessian` loads the original and extended resume states of the
selected unweighted low-rank candidate, then performs short independent continuation
windows in a separate output directory. Original experiment states are not modified.
It does not train GCN evaluation students.

At each outer update the exact inner head is solved once. The same Hessian, right-hand
side, preconditioner and residual tolerance are used for cold (zero) and warm
(previous warm solve) starts. Each condition is repeated with alternating order;
CUDA synchronization brackets each complete solve, including setup and fallback.
The cold solution always drives the assignment update, fixing the common trajectory.
Warm-start failure is recorded, not silently called a speedup; its next initial state
falls back to the cold solution. The first five update pairs are excluded by default.

Reported iteration counts include extended PCG attempts. Residuals and solution
agreement accompany timings. Direct fallbacks may benefit little from warm starts.
The total-step speedup is an estimate replacing only the implicit time on this fixed
trajectory; it is not an end-to-end warm-training measurement. Benchmark wall time
includes both solves and repetitions and must not be presented as normal training
speed. Inner Newton systems are unchanged.

Completed timing windows are cached. Interrupted windows rerun from their original
source checkpoint. Changing source resume metadata invalidates the timing cache.
Numerical tests are provided for Colab; local validation uses syntax and diff checks.
