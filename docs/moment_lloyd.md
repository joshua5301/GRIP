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

For a k-means++-only start, set `initialization_steps=1` and
`require_initialization_convergence=False`: choose centers, assign once (repairing
empty cells if necessary), and compute feature/label means. The converged comparison
uses 300 and True. Both save `initial_center_indices` for paired verification.

`teacher_selection="accuracy_then_ce"` selects gamma by validation accuracy,
breaking ties by validation CE then smaller gamma. Supply placeholder `T=[1.0]`.
After gamma selection, scalar temperature minimizes hard-label validation CE in
inverse-temperature coordinates, bounded by `temperature_bounds=(0.05,20.0)`.
Endpoints and T=1 (when feasible) are checked. The selected temperature replaces
the placeholder before condensation; it is not a student hyperparameter sweep.
Teacher grid, calibration curve, selected temperature and boundary flag are saved.

Initialization-only controls use `max_sweeps=0` (no moment updates) and may use
`lambda=[0.0]`. Use one initialization iteration for selected-center assignment,
or require converged k-means for a standard k-means baseline. Shared runs reuse
only features and gamma-keyed teacher caches, so their coefficient grids and
optimization budgets may differ. Compare variance and moment error separately
when coefficients differ; raw objective values are then not comparable.

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
