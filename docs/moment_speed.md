# Exact variance-moment speed comparison

`src.moment_speed.benchmark_saved_run` reads an existing distance-initialized
sweep's features, CE-selected teacher and initial factors. It never writes to
that source run or changes the production optimizer. Missing initial factors
are built in memory using the same initialization. T is fixed to 1.

All measurements use float64 features, probabilities and factors, the original
variance + lambda moment-norm objective, joint Adam with foreach=False, and
identical initial factors. There is no minibatch approximation, clipping,
annealing, TF32 or lower-precision path. GCN students are not trained.

Variants:

- baseline: existing two-pass block autograd computation;
- direct: fused [mass, feature sums, label sums] with a custom exact backward,
  recomputing probabilities but not cell statistics in backward;
- cached: the same backward with saved full assignment probabilities, blocked
  matrix multiplies and no probability recomputation;
- cached_full: cached custom backward with whole-matrix multiplies;
- full_autograd: whole-matrix probabilities and fused statistics with autograd.

Only U,V require gradients. For material A=[1,X,Q], statistics P.T A/N and
its derivative G, compute D=A G.T/N, then E=P*(D-row_sum(P*D)).
Gradients are dU=E V/sqrt(rank), dV=E.T U/sqrt(rank). Block accumulation
changes floating-point summation order, not the mathematical objective.

Each variant must pass objective and U,V gradient checks at initialization and
after a short baseline trajectory before timing. Failed or OOM variants are
reported, never replaced by another implementation. A CUDA OOM is caught so
other variants can proceed. Tests include double-precision gradcheck and a
small paired Adam trajectory; run them in Colab, not on the local workstation.

Timing uses fresh parameters and Adam state after discarded warmup, randomized
variant order per repeat, CUDA synchronization at the boundaries, and CUDA
events for forward/objective, backward and Adam segments. Per-step finite checks
and best-statistic snapshots are retained. Total wall time includes diagnostics;
event times are approximate phase attribution and may include host-induced GPU
idle intervals. Final objective recomputation, serialization, initialization,
teacher preparation and correctness checks are excluded. Peak allocated/reserved
memory includes shared resident inputs and optimizer state and excludes the
post-timing final objective pass. Projected 3000-step time is an extrapolation,
not an observed full-run measurement.

timings.csv contains raw repeats; summary.csv contains medians and repeat counts.
trajectory_checks.csv compares terminal U,V, J and best sufficient statistics
against the matching baseline repeat. Small errors can grow over a long nonconvex
trajectory; these checks do not establish equality of downstream GCN accuracy.
Completed timing repeats are reused on rerun, and all artifacts live in the
supplied GRIP_results subdirectory. Compare only rows with all repeats completed.
