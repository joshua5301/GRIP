# Initialization comparison

`src.moment_initialization.run_initialization_comparison` reuses the saved
features, teacher logits and selected rank/lambda/gamma/T of a Cora or Citeseer
variance-moment sweep. It validates the source data and feature digests. No
teacher retraining or hyperparameter selection takes place.

The four arms minimize V + lambda ||M|| with joint Adam:

- random: the existing independent standard-normal U,V initialization;
- distance: PCA to at most rank-1 dimensions, seeded k-means++ and 20 Lloyd
  updates, followed by exact factorization of projected squared-distance logits;
- annealed: identical distance initialization, with an explicit entropy reward;
- multistart: lowest unregularized J among independent random runs, including
  the baseline run. Student validation and test never select the restart.

Distance logits are row-centered and scaled to RMS 1 / temperature. Their
rank is at most rank because row constants disappear under softmax. QR/SVD
balances the two factors. The distance construction is used only once: no fixed
distance term, prototype constraint or uniform mixture remains during training.
PCA uses a fixed randomized projection seed shared across condensation seeds.

The annealed loss is J - beta H(P), with beta initially
entropy_fraction * J_initial / log(cells), linearly decreasing to zero at
anneal_fraction * steps. Remaining steps optimize J alone. Default values 0.1
and 0.8 are experimental choices, not theoretically optimal constants. All arms
return their lowest original J checkpoint, including initialization, so annealing
cannot select by its changing regularized objective. History retains every J,
entropy, beta and elapsed time.

Every restart receives the full step budget; multistart costs starts times the
single-run optimization, and its reported seconds sum all restart times. Shared
PCA preprocessing is excluded; distance seeding and factorization are included.
This is not an equal-compute comparison. Numerical timings include diagnostics.

Final students are independent two-layer GCNs trained with uniform soft CE on
the representatives and identity adjacency. All methods use paired student
seeds and the original student settings. Summary seed standard deviations are
over condensation-seed means; by_seed.csv also reports student variability.

Outputs and atomic temporary files stay in the supplied output directory.
Completed condensation trials and student runs are cached, so rerunning the
same cell skips them. Interrupted individual condensations restart from step 0;
there is no optimizer-state resume within a trial. This is a full-matrix small
graph experiment, not a scalable Flickr/Reddit implementation.

## Distance initialization with alternating updates

Set `optimizer_comparison=True` to run distance_joint, distance_alternating,
annealed_joint and annealed_alternating instead of the random/multistart arms.
All four use the same projected features and distance initialization per seed.
`block_steps=50` alternates 50 Adam updates of U with V frozen, then 50 of V
with U frozen. Separate Adam states persist between blocks. Cell moments are
recomputed and differentiated at every update. This is fixed-block coordinate
Adam, not a converged block solve or closed-form ALS; unlike the earlier
moment_alternating experiment, it has no block rollback or early block stopping.
This isolates the update schedule without introducing block acceptance rules
for the changing entropy objective.

Each arm uses the same total gradient-step budget and global-step entropy
schedule. Joint steps update both factors; alternating steps update one, so
per-factor update counts differ. Report wall time as well as update count.
All four return the minimum unregularized J, not the minimum annealed objective.
Completed trials are cached; changing code or settings creates a new run hash.

## SGD comparison

`optimizer_candidates` accepts dictionaries with name, optimizer (`adam` or
`sgd`), lr and optional momentum. This runs distance initialization with joint
updates for every candidate, using entropy_fraction (set 0 to disable annealing).
No weight decay, learning-rate schedule, Nesterov, clipping or gradient rescaling
is applied. SGD here is full-batch gradient descent through the original J,
using PyTorch's SGD optimizer; it is not minibatch stochastic training. The GCN
student optimizer remains Adam and its uniform CE protocol is unchanged.

All candidate results are returned for comparison. This function does not select
an optimizer using test results. Use search student seeds to choose SGD settings
by mean validation across all condensation seeds, then evaluate fixed choices on
separate final student seeds. Candidate names must uniquely encode their settings.
Numerically divergent candidates are recorded in failures.json and excluded from
summary if any condensation seed fails; surviving subsets must not be ranked.
The same immutable run skips recorded failures on rerun. Changed settings create
a different run directory. No silent Adam fallback is used.
