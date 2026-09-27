# Frozen ANIL representation probe audit

This experiment loads the previously selected supervised and ANIL encoders. It
does not retrain them or revisit their checkpoint selection. The original Cora
data and splits are checked against the saved signature. Features are centered
using official train nodes and divided by their RMS centered vector norm. No
validation/test feature statistics are used to fit this normalization.

For each encoder and support size, grid search varies full-batch SGD learning
rate and L2 weight penalty. The objective is mean CE plus penalty/2 times the
squared weight norm; bias is unpenalized. A double-precision L-BFGS polish uses
the same training objective after SGD for every candidate. Thus the learning
rate is an optimization diagnostic: well-solved positive-penalty convex problems
should give similar solutions across rates. No validation early stopping occurs
within a probe. Gradient infinity norm is checked after optimization; a solver
return alone does not establish convergence. Nonconverged runs remain visible
and are not silently described as optimal. Use positive penalties to avoid
unattained unregularized optima for separable supports.

Default model selection minimizes mean validation CE over support/head seeds
0,1,2. Validation accuracy selection is optional. Test metrics are computed only
for the selected settings using fresh paired seeds 100 through 109. Both methods
use identical support sets and initial heads. All head evaluation seeds still
share one already-trained encoder per method. These are not independent encoder
replicates or entirely unseen labels: the encoders used all official train labels.

Saved results include per-trial diagnostics, selected head parameters, support
indices, final per-seed metrics, pre/post-polish objectives and validation CE,
and optimization histories. Feature hashes, code revision and settings determine
cache identity. Source encoder files are never modified. This is a linear probe
experiment rather than an evaluation of graph condensation. No local numerical
tests are run; execute tests/test_anil_probe.py in Colab.
