# Direction-preserving refinement of a saved NTK partition

The Cora study fixes the saved transformed Nyström features, original teacher
probabilities, initial partition, reconstruction protocol and student settings.
The baseline is the saved final partition; the directional method starts from
that exact assignment. Neither condition receives a new hyperparameter sweep.

Train-label linear heads are fitted on class-balanced subsets. Defaults use
70-node supports, penalties .001/.01/.1 and seeds 10/11 for optimization. Audit
heads use 105-node supports, penalties .003/.03 and seeds 200/201. These heads
were not used to optimize the partition; they are not independent datasets or
disjoint train-label pools. Their support sets may overlap. Each fit reports its
gradient and convergence flag; the cached heads and support indices are saved.

For head h let u_ih = W_h^T z_i. The objective is the mean across optimization
heads of (within-cell logit variance)/4 + abs(global logit-label cross-moment
residual). Biases do not enter this objective, but enter exact CE audits. This
is a bound on mean absolute teacher-label risk discrepancy for the fixed heads,
not a uniform bound over all GCNs or all linear classifiers. There is no B or
additional fitted mixing weight. No label or logit normalization is performed.

The solver proposes exact single-node move deltas in blocks and accepts a batch
only if the recomputed objective strictly decreases and every cell stays nonempty.
Rejected batches are recursively split. Stopping requires a full sweep with no
accepted moves; this is not a certificate of a globally optimal partition.

Both final partitions are reconstructed using the same frozen transformed NTK
map and original raw X convex supports, with the original Adam step count/lr.
Reconstruction targets remain NTK feature means, not head logit means. Both
conditions are evaluated with fresh GCNs under identical seeds, mass CE and
validation checkpointing. Test never selects a head, partition or setting.
Kernel-feature CE diagnostics exclude reconstruction; student metrics include it.

Saved outputs: partition history, both reconstructions, per-head pre/post bounds
and exact CE gaps on ideal centers, head convergence records, student results,
and configuration/source identifiers. The original cache is read-only.
Numerical tests run only in Colab; local checks are syntax and diff checks.
