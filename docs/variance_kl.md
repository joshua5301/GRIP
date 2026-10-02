# Variance + label KL control

Use `src.variance_moment_sweep.run_risk_sweep(method="variance_kl")`.
The gamma/T/B grid, normalized S²X, historical seed partition, condensation seeds,
teacher and uniform-CE two-layer GCN evaluation match the variance-moment runner.
This is squared feature distance, not the original unsquared GRIP objective.

Let V be mean within-cell squared feature distance, K be mean forward
KL(q_i || s_j), and M the global feature-label cross-moment discrepancy.
Features and labels use arithmetic cell means. Pinsker and Cauchy-Schwarz give
`||M||_F <= sqrt(2 V K)`. For the historical implementation's coefficient 2B,
Young's inequality yields

`B² V/4 + 2B ||M||_F <= B² V/2 + 8 K`.

We optimize the right-hand side, without another hyperparameter. Its equivalent
distance-normalized KL weight is `16/B²`; B is not a direct KL weight. Equal B
preserves the initialization and sweep count, not the effective trade-off or
objective value. This bound does not imply higher GCN accuracy. It inherits the
linear-head setting and any representation assumptions of the original bound.

Chunked Lloyd assignments minimize `B²/2 ||x_i-c_j||² + 8 KL(q_i||s_j)`.
Both centers and labels update to arithmetic means. Empty cells are reseeded
with a node from a nonsingleton cell, preserving the budget. That operation
cannot increase the objective: the new singleton has zero cost, and re-averaging
minimizes the remaining cells' cost. Exact ties retain existing assignments.
Zero-probability supports use infinite assignment cost for positive q against
zero s. Convergence means assignments no longer change, not merely small progress.

`shared_run` may point to a historical sweep's fingerprint directory. Relevant
data, teacher, grid, seed and evaluation settings must match. Features and any
completed teacher caches are copied into the new run so both methods use the
same inputs. Missing teachers are fitted normally. Never share an actively
written candidate directory; student and partition caches are separate.

`partitions.csv` records partition seconds and KL; each partition contains its
initial assignment digest and objective history. `search_students.csv` reports
wall time for freshly evaluated students and marks cache hits (time is NaN).
KL partition timings synchronize CUDA; the historical solver's timing does not
explicitly synchronize, although its many scalar checks synchronize internally.
Compare matched candidate/seed partition times, not only each method's winner.
Student fits and full-graph validation costs remain unchanged and can dominate.

Tests are provided for Colab in `tests/test_variance_kl.py` and the shared sweep
tests. No numerical tests or training are run locally. Changes to runner code
create new fingerprints, retaining previous results without automatic migration.
