# Moment-corrected Lloyd proposals

The objective is `variance + lambda * ||global feature-label residual moment||_F`.
`moment_lloyd_hybrid`, `moment_lloyd_full_only` and `moment_lloyd_filtered_batch` use identical seeded
k-means++ followed by ordinary Lloyd iterations on the full centered,
RMS-normalized S²X features. There is no PCA, low-rank factor, annealing or GD.
This initialization differs from the projected initialization of the low-rank method.
`initialization_steps` defaults to 20; set 300 and `require_initialization_convergence=True`
to require unchanged assignments before optimization. Hitting the initialization limit
then raises an error. Artifacts store the initialization iterations and convergence flag.

With feature and label means c,s, and G=M/||M||, assignment proposals minimize
`||x-c||² + lambda*(x-c)^T G (q-s)`. This is a first-order cost, not an upper bound.
At zero M the implementation chooses zero as a norm subgradient.
Every accepted update is checked using the original objective and new cell means.
Empty-cell proposals are rejected; ties retain the old assignment.

Full-only stops at the first rejected full proposal (or no proposal). This is not
a certificate of single-node stationarity. Hybrid then checks decreasing prefixes
of moves ranked by predicted gain. If these fail, it scans exact single-node
relocations and accepts the first verified improvement, then rebuilds proposals.
The scan forbids deleting singleton cells. A complete scan with no accepted move
is stationarity to numerical tolerance. An iteration limit is not convergence.

Filtered-batch uses the same full proposal followed by top-gain prefixes (half,
quarter, etc.) but never scans exact single moves. `backtrack_steps=8` limits the
number of partial attempts for both filtered-batch and hybrid. Rejection or no
proposal stops filtered-batch without declaring stationarity. Records include
candidate count, accepted move fraction, and objective evaluation count.

Artifacts retain the initial assignment digest, original objective history,
termination status, and per-attempt kind, check count and accepted move count.
The exact scan can be slow: speedup is not assumed. Use timing and acceptance
records alongside validation accuracy. Student evaluation remains uniform CE.

Run tests/test_moment_lloyd.py in Colab. Tests cover the derivative, exact relocation
formula, paired initialization, monotonic acceptance and rejection behavior.
