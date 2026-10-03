# Moment-corrected Lloyd proposals

The objective is `variance + lambda * ||global feature-label residual moment||_F`.
Both `moment_lloyd_hybrid` and `moment_lloyd_full_only` use identical seeded
k-means++ followed by at most 20 ordinary Lloyd iterations on the full centered,
RMS-normalized S²X features. There is no PCA, low-rank factor, annealing or GD.
This initialization differs from the projected initialization of the low-rank method.

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

Artifacts retain the initial assignment digest, original objective history,
termination status, and per-attempt kind, check count and accepted move count.
The exact scan can be slow: speedup is not assumed. Use timing and acceptance
records alongside validation accuracy. Student evaluation remains uniform CE.

Run tests/test_moment_lloyd.py in Colab. Tests cover the derivative, exact relocation
formula, paired initialization, monotonic acceptance and rejection behavior.
