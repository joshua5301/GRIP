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

## Matched low-rank hyperparameter extension

`run_low_rank_shard` reuses this sharded runner with joint feature/label/mass updates
from a fixed-rank assignment. Node weights remain one; no weighted-node extension
or implicit-solver benchmark is enabled. Teacher logits, features and initial
assignments are loaded from the original family sweep. Each new candidate starts
from the original initialization and resumes only its own checkpoint if interrupted.
The output fingerprint separates low-rank and C-mean runs. Existing C-mean paths
and optimizer configuration remain compatible.

Suggested Arxiv rank-eight grid: T = [0.3, 0.5, 1.0], inner penalty = [1e-5, 3e-5,
1e-4], assignment learning rate = [0.01, 0.03], 1000 steps. Eighteen candidates are
split into six per session. Validation chooses both hyperparameters and checkpoint;
only the final selected candidate receives test evaluation on disjoint student seeds.
