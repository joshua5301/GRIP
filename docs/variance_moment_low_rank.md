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
