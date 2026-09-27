# From fixed-head risk to newly learned heads

The experiment reuses saved baseline/directional NTK partitions and raw convex
representatives. A new regularized linear softmax head is trained on each ideal
NTK centroid set, and separately on the actual mapped synthetic inputs. Both
use cell-mean original teacher probabilities and cell-mass CE. Evaluation uses
the original graph's node features in the same frozen feature space.

The full-data reference trains on original-node teacher probabilities, not just
140 true training labels. Its regularization coefficient is selected by official
validation CE from a fixed grid, then held fixed for all four partition/stage
conditions within that representation and density. No test metric selects a
coefficient. Test is evaluated only for that selected coefficient. All training
objectives use the same weight penalty and unpenalized bias. Float64 L-BFGS
starts at zero; gradient diagnostics are reported rather than assuming convergence.

`full_objective` evaluates each learned head on all original teacher-labeled
nodes with its weight penalty. `excess_full_objective` subtracts the optimized
full-data reference. This directly checks whether training on compressed data
produces a good solution of the original surrogate learning problem. It is not
a uniform-risk or ground-truth generalization certificate.

The optional S²X control uses the RMS NTK condition's original gamma, T, B and
student settings for a matched-label diagnostic, not a separately tuned S²X
best result. It recomputes a variance-moment partition in S²X, then refines that
same assignment with the directional objective using the prior study's head
support sizes, penalties and seeds. The classifier feature space is centered/RMS
S²X. Synthetic GCN inputs are raw S²X cell means with A=I; applying the same fixed
centering/scaling to them reproduces the ideal centers up to float precision.
Thus ideal/realized linear probes should agree for S²X. This exact realization
is an SGC-feature statement, not an equivalence with nonlinear GCN computation.

Fresh GCNs compare the two S²X partitions under matched seeds and mass CE. NTK
GCNs are not retrained; their preceding results remain the relevant comparison.
Saved outputs include per-stage learning metrics, full-reference selection grid,
linear heads, S²X partitions and per-seed GCN metrics. The preceding cache remains
read-only. Tests run in Colab only; local checks are limited to AST and diff checks.
