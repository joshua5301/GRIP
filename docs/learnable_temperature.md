# Learnable condensation temperature

The teacher logits a remain frozen. A single scalar t = log(T) is trained with
the low-rank assignment factors. The condensed labels use q_i(T) = softmax(a_i/T)
and the usual mass-weighted cell averages. Features and cell masses retain their
existing definitions. No extra label parameters or node weights are introduced.

The input q to optimize_ce_assignment is the fixed outer reference, not a mutable
temperature-dependent target. In the comparison runner, both methods use the source
temperature (currently 0.3) for this reference and for their initial labels. Only
the learned method subsequently changes the inner labels. The loss normalization
is still the initial outer CE and is identical at initialization.

With g = gradient_W L_inner and v = H^{-1} gradient_W L_outer, the temperature
hypergradient is -v^T partial_t g, computed through the existing moment gradient.
The implicit solve is shared with assignment updates. Both cached and recomputed
low-rank moment backward functions now propagate gradients to their material input
when required. This introduces extra matrix multiplication and gradient storage,
but no additional Hessian solve. Frozen-material runs keep the original path.

Temperature uses its own Adam group (default learning rate 0.003), T=exp(t),
without clipping or a temperature regularizer. Nonfinite/nonpositive values fail
explicitly. Very small or large finite temperatures can still saturate the labels;
inspect the saved temperature and gradient trajectories. The teacher temperature
gradient respects the original teacher-logit dtype, then labels are cast to double
as in earlier experiments.

The implementation currently supports exact unweighted node-factor optimization,
free cell masses, and jointly formed features/labels. Resume files include the
temperature parameter, its Adam state, teacher-logit digest and learning rate.
Checkpoints store the current temperature and materialized representatives.

compare_temperature reruns fixed_T and learned_T from the same cached source
initialization. Both use Newton-first and implicit warm starts, with optional
grouped PCG and probability caching. Hyperparameters other than T remain fixed.
Checkpoint selection uses GCN validation with three search seeds. Fixed-budget and
selected-checkpoint evaluation use ten separate paired seeds, two-layer GCN,
identity synthetic adjacency and mass CE. Test results do not select checkpoints.
Original experiment artifacts are not modified. Changing temperature_lr creates a
separate output directory; interrupted runs resume at their own saved checkpoint.

Tests cover the material/temperature derivative against ordinary autograd, the
implicit hypergradient against centered finite differences of independently solved
inner problems with a fixed outer target, and temperature restoration on resume.
Run these numerical tests in Colab; local checks are syntax-only.
