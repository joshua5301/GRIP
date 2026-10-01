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
edge-preserving low-rank coarsening. `distance` learns positive, mean-one diagonal
distance weights from feature pairs. Its fixed metric anchors and baseline logits
preserve the initial assignment exactly; it changes P while keeping representatives
as means in the original normalized H space. The `nystrom` method keeps low-rank assignments
and replaces the linear surrogate input by the teacher's explicit Nyström map,
applied after forming original-H centroids; it still fits CE rather than MSE.
Original-node kernel features are cached on CPU. Initializers are feature k-means,
`teacher_joint` (RMS H concatenated with alpha times centered teacher Q), and
`teacher_balanced` (teacher-inferred class quotas). They use no unlabeled-node
ground-truth classes. Features and labels remain assignment-derived means.
The optional `train_target_mix` ablation mixes known training labels into teacher
probabilities on training nodes only. Its defaults preserve the original teacher
targets; held-out labels never enter this mixture or its initializer.

Coarsening forms original-feature centroids Xc=PᵀX/m and normalized quotient
Sc=normalize(PᵀSP), optionally with cluster-mass scaling. S is the existing
normalized original adjacency including self loops; quotient diagonal is retained.
The exact CE surrogate uses Sc²Xc, with a fixed original-H RMS transform. Its
implicit derivative includes both the centroid and edge paths. Final students
train on Xc,Sc, rather than adding more propagation to already propagated H means.

Final student CE is uniform. Surrogate CE supports `inner_loss_weighting` mass
and uniform. Low-rank/MLP candidates can additionally set `mass_mode="uniform"`
to constrain P's column masses with the existing differentiable balancing solver.
This changes the partition; it is separate from the CE weighting. Other assignment
methods reject this option. Student input scaling is applied identically to synthetic and original
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

## Sustained local runs

`src.research_loop` executes `results/research_loop/plan.json` serially on the GPU.
Each concrete job has a wall-clock limit, and its immutable ID/configuration and
outcome are recorded in `jobs.json`. The thread follow-up reviews completed batches
and adds the next plan. An exhausted queue is `awaiting_next_plan`, never proof
that the research objective was achieved. The current plan prioritizes all six
Cora/Citeseer budgets and includes bounded Arxiv/Flickr/Reddit pilots.

```bash
.venv-local/bin/python -u -m src.research_loop results/research_loop/plan.json
```

Create `results/research_loop/STOP` to stop the worker at its next safe check.
Remove it only after a user-authorized restart. Keep the computer awake and the
desktop app running for the scheduled thread follow-up. `src/local_experiment_loop.py`
is an earlier unused prototype; the bounded queue is the current runner.

`large_pilot.run_pilot` uses a shared streamed Nyström teacher and validation-only
GCN evaluation on each split's own graph. Its initial 512-anchor pilot is explicitly
recorded and is not interchangeable with the 3,000-anchor benchmark. Final student
CE remains uniform. Arxiv training diagnostics use only its public training mask.

Nyström feature caches now verify H, the full map and cached features. Legacy
feature caches migrate only after complete coordinate verification; unverified
legacy resume files are preserved and rejected. Student caches verify both inputs
and recipe, preserve legacy scores before retraining, and score test only once
after restoring the validation-selected weights. All result claims still require
matched protocols and independent condensation repeats.
The original propagated H and complete Nyström map are frozen per data/processing
provenance, so repeated sparse propagation and Cholesky calls do not produce
different coordinates across condensation seeds. New citation maps use schema 3;
older coordinate caches and resume artifacts are preserved separately.

Large teachers load only training rows of phi onto the GPU when the measured free
memory permits; the all-node cache stays on CPU and prediction remains streamed.
This evaluates the same CE objective while avoiding repeated PCIe transfers.
Final citation tables additionally report MLP and GCN serving with the identical
GCN-validation-selected weights. This secondary comparison does not retrain or
choose a test epoch. Change code only between worker invocations; fresh workers
record a source manifest and Git revision for their jobs.
Low-rank/MLP optimization also checks the worker stop request between solver calls;
checkpoint files are replaced atomically so an interrupted write cannot masquerade
as a completed endpoint.

`student_settings` on `run_screen` and `selected_test` can override `lr`,
`weight_decay`, `eval_every`, `hidden`, `lr_schedule`, and `initialization`.
The historical default is PyG initialization and Adam reset to lr/10 halfway
through training; omitting overrides preserves existing cache identifiers.
`lr_schedule="constant"` and `initialization="geom_uniform"` reproduce GEOM's
constant learning rate and its uniform fan-in initialization, including biases.
Validation still selects the epoch and test is scored only after selection.
GEOM Cora-70 uses row-normalized features, 600 epochs, dropout 0, lr .001,
weight decay .001, and validation every epoch. Record those choices together;
changing only the evaluator on a raw-feature condensate does not reproduce the
row-normalized source protocol. These controls align student settings, while
our assignment-derived soft labels remain part of the condensation method.
