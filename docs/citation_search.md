# Short local clustering experiments

Local numerical training is authorized for the current research goal. Use the
existing `.venv-local` Python environment. `run_screen` runs Cora/Citeseer in
small resumable rounds; 100 updates on this RTX 4060 took about 7–9 seconds per
candidate including the first two validation students. This is a measurement for
these citation graphs, not a large-dataset runtime prediction.

The teacher fits only public training labels. Every candidate/checkpoint is
selected using validation. `selected_test` evaluates one fixed configuration and
checkpoint with separate student seeds. Result tables record condensation seeds
separately; student fits are not independent condensations. Repeated research
rounds do not imply a statistically verified SOTA result.

Supported assignment variants are node low-rank factors, a feature MLP, and
edge-preserving low-rank coarsening. The `nystrom` method keeps low-rank assignments
and replaces the linear surrogate input by the teacher's explicit Nyström map,
applied after forming original-H centroids; it still fits CE rather than MSE.
Original-node kernel features are cached on CPU. Initializers are feature k-means,
`teacher_joint` (RMS H concatenated with alpha times centered teacher Q), and
`teacher_balanced` (teacher-inferred class quotas). They use no unlabeled-node
ground-truth classes. Features and labels remain assignment-derived means.

Coarsening forms original-feature centroids Xc=PᵀX/m and normalized quotient
Sc=normalize(PᵀSP), optionally with cluster-mass scaling. S is the existing
normalized original adjacency including self loops; quotient diagonal is retained.
The exact CE surrogate uses Sc²Xc, with a fixed original-H RMS transform. Its
implicit derivative includes both the centroid and edge paths. Final students
train on Xc,Sc, rather than adding more propagation to already propagated H means.

Final student CE is uniform. Surrogate CE supports `inner_loss_weighting` mass
and uniform. Student input scaling is applied identically to synthetic and original
features. It changes the optimization/regularization recipe, not labels or splits.
`citation_features="raw"` preserves raw Planetoid features. The default loader
still row-normalizes Citeseer and retains raw Cora, preserving previous runs.
Every preprocessing choice is fingerprinted through input data and configuration.

Example round (Python, after local environment setup):

```python
from src.citation_search import run_screen

ranking, root = run_screen(
    "citeseer", 0.018, "results/citation_search_v1",
    [dict(T=0.3, rank=8, penalty=1e-3,
          initialization="teacher_joint", alpha=0.3)],
    steps=100, epochs=1000, dropout=0.5,
    citation_features="raw", data_dir="data",
)
```

The root stores dataset/teacher provenance, candidate definitions, every screen
protocol and student recipe, resumable factors, checkpoints, validation-only
tables and fixed-selection final tables. For a common multi-seed selection,
aggregate the same candidate and checkpoint across all search condensation seeds;
do not choose a different candidate or step per seed. Published scores should be
matched by actual node count and allowable label access, not percentage alone.
