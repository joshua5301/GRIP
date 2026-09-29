# C-mean control against assignment families

`run_cmean_shard` optimizes only synthetic features. Cell labels and mass CE weights
remain fixed at initialization. It loads the source assignment-family sweep's cached
RMS features, teacher logits, k-means partition, GCN settings and seed lists. Gamma is
not retuned. Source graph and split digests are checked. Initial moments at shared
temperatures are compared with the source cache. The initializer uses rank-eight
zero correction to avoid allocating dense scores; rank is not a learnable control
parameter, because direct optimization replaces those parameters with cell features.

The temperature/penalty/learning-rate Cartesian grid is split by candidate index
modulo the shard count. Every candidate gets the same full optimization budget;
all saved steps, including zero, compete on mean GCN search validation accuracy.
Each shard has separate checkpoints, Adam resume states and student caches.
After every shard completes, rerun the final shard's cell to aggregate if necessary.
Only the last numbered shard writes aggregate results, preventing concurrent final
cache writes. Completed optimization and student evaluations are reused.

The global validation-selected candidate is evaluated on the source's disjoint final
student seeds. Test results are not used for selection. Original family outputs are
never modified. The control uses a broader search than the original family sweep;
report this asymmetry rather than claiming equal tuning budgets. Independent
condensation-seed repetitions remain a separate experiment.

Local checks: syntax parsing and diff checks only; training is reserved for Colab.
