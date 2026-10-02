# Low-rank variance–moment optimization

Use `run_risk_sweep(method="variance_moment_low_rank")`. The teacher, normalized
S²X and seed partition are inherited from the historical variance–moment runner.
No student solve, Hessian solve or bilevel differentiation is used in condensation.

For hard initialization H, P0=(1-mixing)H+mixing/K and
P=softmax(log(P0)+UVᵀ). Defaults: rank 8, mixing .05, Adam LR .01, 1000 updates.
U is Gaussian with standard deviation 1/sqrt(rank); V starts at zero. Thus the
initial P equals P0 exactly, with a nonzero gradient available to V. The fixed
base logits are generated per chunk from the hard assignments, not stored dense.
Mixing is a numerical modeling choice, not derived from the risk bound.

Let pi=Pᵀ1/N, F=PᵀZ/N, L=PᵀQ/N. Centers and labels are F/pi and L/pi.
Optimize B²/4 [mean(||z||²)-sum(F²/pi)] + 2B ||ZᵀQ/N-Fᵀ(L/pi)||_F.
The norm used for gradients is sqrt(norm²+1e-24)-1e-12; selection and logs use
the exact norm. This is a soft relaxation of the historical hard objective.
The output uses soft representatives without hardening and preserves total
feature and label means under cell mass weights. Final GCNs still use uniform CE.

Each update accumulates detached sufficient statistics by chunks, differentiates
the small statistics objective, then recomputes each chunk to backpropagate its
contribution. This is the full chain-rule gradient, not a detached-center update.
Memory avoids retaining the full N-by-K assignment autograd graph; computation
still scales with N K (rank + feature dimension + classes).

All steps are logged. The representative statistics at the lowest exact J,
including step zero, are returned. GCN validation selects gamma/T/B only after
condensation; it does not select optimization steps. `best_step` is in partitions
and by-seed CSVs. `sweeps` means optimizer updates for this method. A fixed budget
does not prove stationarity, so `converged=False` and status=fixed_step_budget.
Completed runs are cached; an interrupted individual partition restarts.

The initialization is identical to the historical hard partition before mixing,
but its soft objective at step zero is not the historical hard initial objective.
Rank, mixing, learning rate and step budget are fixed controls, not swept by the
gamma/T/B grid. Tests are intended to run in Colab, not locally.

Set `assignment_initialization="random"` to remove the historical seed partition
and all fixed logits. Independent standard-normal U and V produce
P=softmax(UVᵀ/sqrt(rank)). Mixing is unused. The condensation seed controls only
factor initialization; it is independent of gamma/T/B. Supply `rank` as an
additional grid key to select gamma/T/B/rank jointly by mean GCN validation.
The 1/sqrt(rank) scaling standardizes initial logit variance, not optimization
dynamics. The historical initialization remains available as the default.

For random initialization, replace grid key `B` with `lambda` to optimize
`variance + lambda * global_moment_norm` directly, including in gradients and
checkpoint selection. `lambda=8/B` maps the old objective up to a positive overall
factor, preserving minimizers but not necessarily finite-step Adam trajectories.
The lambda and B grids are mutually exclusive. No label-variance normalization
is introduced; the existing centered, RMS-normalized feature space is retained.

`assignment_backend="auto"` uses a single full-assignment autograd pass when
N*K <= 2,000,000 (including Cora/Citeseer), and the two-pass chunked calculation
otherwise. Explicit `full` and `chunked` overrides support timing comparisons.
Full avoids recomputing probabilities and sufficient statistics for backward.
It preserves the objective, float64 arithmetic and optimizer, apart from floating
point accumulation differences. This threshold is a heuristic, not a GPU memory
guarantee. Nonfinite checks aggregate device flags before host synchronization.
Tests compare both backends' trajectories and representatives in Colab.

`teacher_selection="validation_ce"` requires T=[1.0]. Fit every requested gamma
on training labels, select the lowest hard-label validation CE (smaller gamma
breaks exact ties), then condense only that gamma's rank/lambda grid. The teacher
choice is saved in selected_teacher.json before any condensation. Teacher accuracy
is diagnostic only and no test labels are used for selection. Default `grid`
continues to tune gamma jointly with condensation parameters. Teacher preselection
and student selection share validation, so validation is not an independent holdout
estimate. Final test is evaluated only for the selected condensation setting.

`evaluate_test=False` saves search tables and selected.json, and returns empty
summary/by-seed frames without final students. Repeating the same run with True
reuses its search/partition caches and evaluates its winner. `candidate_subset`
accepts complete grid dictionaries for shortlisted settings; it is fingerprinted.
This supports validation-only screening at 1000 updates followed by 2000/3000
updates on the top settings. Use the initial run as shared_run to retain exact
features and teacher logits. Larger budgets currently restart from the same seed;
optimizer state continuation is not implemented. Compare stage winners by search
validation and invoke final evaluation only on the winning stage.
# Distance initialization across datasets

The low-rank sweep grid also accepts `assignment_lr`, e.g. [0.01, 0.1].
It overrides the fixed assignment_lr argument for each candidate and is retained
in search, selected and final tables. Student learning rate remains independent.

`run_risk_sweep(..., assignment_initialization="distance")` supports the lambda
objective on all configured datasets. It uses the same PCA, k-means++ seeding,
20 Lloyd updates and balanced distance-logit factors as moment_initialization.
Distance temperature is fixed at 1, with row-centered logits scaled to unit RMS.
Only the initial factors encode distance: training optimizes free U,V jointly
with Adam, without annealing or a fixed distance bias.

Lloyd assignments are computed in blocks. The existing auto backend computes
soft-assignment moments and gradients in blocks when N*K exceeds 2,000,000.
This avoids storing the whole assignment matrix, but does not reduce the
O(N*K) work per step. The blockwise gradient is the full objective gradient,
not a minibatch approximation. Runtime can remain substantial on large graphs.

Within each fingerprinted sweep directory, PCA projections are cached by rank
and initial factors by rank/cell count/seed, shared across lambda values. Cache
reuse makes partition_seconds exclude already cached preparation. Do not reuse
an initialization cache directly with different input features. Source/data
fingerprints isolate caches in the sweep API. Completed trials resume; interrupted
individual condensations restart. All cache and temporary files stay under the
configured output directory.

