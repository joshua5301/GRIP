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
