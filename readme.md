# Learnable clustering for graph condensation

Learn a soft assignment from graph nodes to condensed cells. Features and labels
are assignment-weighted means; the condensed graph uses identity adjacency.

The current protocol uses S짼X features, a kernel teacher, cell-mass-weighted inner
CE and full-node outer CE. A linear inner student is solved with Newton-first
optimization and implicit differentiation. Final evaluation uses a two-layer GCN
with uniform CE. Teacher gamma, condensation settings and checkpoints are selected
using validation only.

## Code

| Module | Purpose |
| --- | --- |
| `src/multiseed_sweep.py` | Cora grid search across condensation and student seeds |
| `src/soft_ce_partition.py` | Bilevel CE optimization, implicit solves and resumable checkpoints |
| `src/low_rank_assignment.py` | Low-rank factors, linear/MLP encoders and optional node weights |
| `src/moments.py` | Dense soft assignments and weighted cell statistics |
| `src/balanced_assignment.py` | Optional equal-mass assignments |
| `src/head.py` | Regularized linear CE head |
| `src/teacher.py` | Kernel teacher and validation-based gamma selection |
| `src/data.py` | Cora, Citeseer, Arxiv, Flickr and Reddit loading and propagation |
| `src/initialization.py` | Seeded k-means++ initialization |
| `src/transforms.py` | Feature normalization |
| `src/evaluation.py` | GCN training and split metrics |
| `src/sweep_utils.py`, `src/io.py` | Grid utilities and persistence |
| `src/models.py`, `src/solver_benchmark.py` | GCN and paired inner-solver timing |

## Colab

After mounting Drive, cloning the repository and preparing data, use one cell:

```python
%cd /content/GRIP
!git pull --ff-only origin main

from src.multiseed_sweep import run_cora_multiseed

summary, by_seed, search, root = run_cora_multiseed(
    ratio=0.013,
    output_dir="/content/drive/MyDrive/GRIP_results/cora_lowrank_gamma001_ratio0013_v1",
    gammas=[0.01],
    space={
        "T": [0.3, 1.0, 3.0],
        "rank": [8, 16, 32],
        "penalty": [1e-5, 1e-4, 1e-3, 3e-3, 1e-2],
    },
    steps=1000,
    checkpoint_steps=[0, 100, 300, 500, 750, 1000],
    condensation_seeds=[0, 1, 2],
    search_seeds=[0, 1, 2],
    final_seeds=list(range(100, 110)),
)
display(summary, by_seed)
```

Each setting/checkpoint is scored by nine validation evaluations (three
condensation seeds 횞 three student seeds). Final evaluation uses ten student seeds
per condensation seed. The three condensation-seed means, their sample standard
deviation, and within-condensation student variability are reported separately.

The existing `run_cora_multiseed` interface, run fingerprints and joint-clustering
checkpoint format are retained. Identical settings and output paths reuse completed
work and resume interrupted optimization. After updating an already-imported runtime,
restart it and rerun the same cell to load all relocated modules consistently.

Legacy GRIP, risk-bound clustering, NTK, ANIL, tree-distance, trajectory and free-feature
control experiments have been removed. Their old notebook imports are no longer
supported; previous revisions remain available in Git. Dense, low-rank and
feature-conditioned assignment optimization remain available through
`optimize_ce_assignment`. Direct feature controls are rejected explicitly.

See [the sweep protocol](docs/multiseed_sweep.md) for selection and caching details.

## Verification

Run numerical tests in Colab, not on the local workstation:

```python
%pip -q install pytest
!python -m pytest -q tests
```

Local refactoring checks cover syntax, static imports and lint only. Numerical
training equivalence must still be verified in Colab.

## Historical comparison

`src/risk_reproduction.py` runs the requested fixed-setting Cora replay and a
2×2 initialization/optimizer comparison, loading pinned source from Git history.
See [the reproduction protocol](docs/risk_reproduction.md).

Prototype and feature-only MLP sweeps are also supported by `run_cora_multiseed`
with `method="prototype"` or `method="mlp"`; see the sweep protocol for grid keys.

`src.teacher_student_selection.select_cora_teacher` selects gamma and temperature
before condensation by validation CE of a converged linear student on all original
RMS-normalized S²X features and teacher soft labels. The kernel teacher uses only
training labels; validation labels are used for selection, never student fitting.
The selection student's penalty defaults to 0.001 and remains independent of the
subsequent condensation penalty sweep. The returned gamma and T can be passed as
singletons to `run_cora_multiseed` for each density. Selection grids are cached and
resumable; use separate output folders for concurrent sessions. Test labels are
not used in selection. This selects over the supplied finite temperature grid,
not continuous temperature optimization or cross-validation.

`select_cora_teacher_ce` in the same module instead selects gamma by the teacher's
hard-label validation CE after temperature calibration for each gamma. Temperature
is optimized continuously in log space within configurable bounds (0.01, 100 by
default); boundary selections are reported. No auxiliary student is trained.
Pass the returned gamma and T as singleton grids to the condensation sweep.

For a teacher-free control, call `run_cora_multiseed(label_source="train",
gammas=[], space={"rank": [8], "penalty": [...]}, ...)`. Only training nodes and
their one-hot labels enter condensation and its inner/outer objectives. S²X and
its RMS transform still use the full transductive graph, matching the teacher
baseline. Representatives are reconstructed from training-node features; final
two-layer GCN evaluation still uses uniform CE. There is no temperature grid or
teacher fitting, and teacher-specific evaluation metrics are omitted. The default
`label_source="teacher"` preserves the existing protocol and cache fingerprints.

For sparse assignments use `method="sparse"` with grid keys `T`, `k`, `penalty`.
Each node retains its initial k-means cell plus the nearest k-1 initial centroids
in RMS-normalized S²X. Candidates remain fixed. Trainable N-by-k logits use row
softmax; other cells have zero probability. Initial mass is 0.95 on the original
cell plus 0.05 uniformly over the k candidates. Features, labels and cell masses
use these probabilities. Sparse moments and their custom backward avoid an N-by-M
allocation during optimization; initialization uses chunked centroid distances.
This changes both the support constraint and initial soft assignment relative to
the dense/low-rank baseline. Rank and assignment temperature are not parameters.
To tune teacher gamma by condensation performance, call the sweep separately for
each singleton gamma and select the global winner by search validation only.
Run `tests/test_sparse_assignment.py` in Colab for gradient and resume checks.

`src.soft_init_sweep.run_soft_init_sweep` implements a two-stage Cora experiment.
First, fixed k-means centers define P=softmax(-squared_distance/tau) in the existing
RMS-normalized S²X space, without uniform mixing or soft Lloyd iterations. Gamma,
teacher T and assignment tau are jointly selected by mean uniform-CE GCN validation
accuracy across condensation and student seeds. All teacher candidates are used;
teacher accuracy does not preselect gamma. Then only the selected initialization
is fine-tuned using fixed distance logits plus UVᵀ/sqrt(rank), mass-weighted inner
CE and the original implicit solver. Inner penalty and step (including zero) are
selected by validation. Independent final student seeds evaluate initial and
selected checkpoints on test. Initial selection and fine-tuning share exact cached
inputs and teacher logits; their step-zero moments are checked for agreement.
Temperature tau is in normalized squared-distance units, not teacher temperature
units. A larger initial grid costs GCN training but no bilevel optimization.
Use an unchanged output directory/configuration to resume. Numerical checks live
in `tests/test_soft_init_sweep.py` and must run in Colab, not locally.

`src.soft_init_sweep.run_distance_cost_sweep` uses row-centered squared distances
divided by their global RMS: Dn=(D-row_mean(D))/RMS(D-row_mean(D)). Assignments are
P=softmax(-(Dn+UVᵀ/sqrt(rank))/t). Centers and distance normalization stay fixed.
U starts at zero and V at unit Gaussian scale, preserving the initial assignment
exactly. This measures the learned cost in distance units and places both costs
under the same temperature; the correction need not remain a metric.
Gamma and teacher T are selected by initial condensed GCN validation at reference
t=1. They are then fixed while assignment t, inner penalty and checkpoint are
selected by validation. Rank defaults to 8. Inner CE is mass weighted; evaluation
uses fresh two-layer GCNs with uniform CE. Three condensation seeds and disjoint
search/final student seeds are used. Test is evaluated only after selection.
The summary separates selection_initial (reference t=1), initial (selected t,
before learning), and selected (selected t, after learning). Compare the last two
to isolate optimization gains. The returned initial grid calls reference t `tau`
for compatibility; the fine-tuning grid and summary call it `t`. This sequential
selection does not jointly optimize gamma, T and t. Reusing the same configuration
resumes optimization and cached student evaluations. The old unnormalized sweep
retains its original formula and cache identity.

For the matched random-prior ablation, set `initialization="random"` in
`run_distance_cost_sweep`. No k-means is called. A seeded N-by-M Gaussian cost
matrix, independent of features and labels, replaces D; identical row centering
and RMS scaling apply. U=0, Gaussian V, rank, temperature and all training rules
remain unchanged. The dummy assignment only specifies the number of cells; it
does not define probabilities, representatives or labels. The random base remains
fixed during optimization, so this ablates the geometric prior as well as the
starting assignment, not solely an initial value in a shared parameterization.
For Cora 1.3%, compare both priors at gamma=0.0001, T=2, t=0.3, penalty=0.003,
rank=8 and lr=0.01, with the same checkpoints and seeds. This is a matched-setting
ablation, not a separately tuned performance comparison.

Pass `resume_from` with an existing run directory to extend the cost sweep.
Only the step budget and future checkpoints may change. The runner copies the
original run into the new configuration directory, preserving the original files,
and resumes every candidate's parameters, Adam state and solver warm starts.
Retain all old checkpoints. Selection considers both old and new checkpoints.

For a fresh comparison across methods, pass the same `shared_features_path` to
each run. The first run saves S²X and a digest of the original graph features,
labels and masks; later runs reuse exactly those propagated features after
checking the raw graph digest. This avoids separate sparse propagation results
changing the experiment input between methods.

Set `initialization="none"` in `run_distance_cost_sweep` to remove the fixed
cost entirely: P=softmax(-UVᵀ/(sqrt(rank)*t)). Both factors start from seeded
unit Gaussian values. Zero U would make all initial cells identical, so this
initialization breaks that symmetry. The same rank, temperature, inner model,
optimizer and evaluation apply. Use the shared propagated features from the
k-means/random-cost ablation for a matched Cora comparison.

`src.soft_init_sweep.run_dual_mlp_width_sweep` uses two independent two-layer
feature MLPs: U_i=g(H_i) and V_j=h(H_{a_j}), where a_j is one of M seeded,
distinct random original-node anchors. Each network has one ReLU hidden layer
and an output of the requested rank. Only node features enter either network;
anchors stay fixed and no k-means or fixed logit cost is used. Input scaling
restores per-coordinate variance after RMS normalization. The assignment is
softmax(-UVᵀ/(sqrt(rank)*t)). The sweep selects width and checkpoint by mean
GCN validation over the same condensation/student seed pairs, then evaluates
only the selected setting on independent student seeds. `t` and inner penalty
are fixed by the caller. The complete grid and optimizer states are resumable.

`run_distance_cost_sweep` also supports inductive Reddit with `dataset="reddit"`
and ratio `0.001` (153 representatives). Teacher gamma can be selected once by
inductive validation accuracy with `teacher_selection="accuracy"`; subsequent
initial and fine-tuning settings use uniform-CE two-layer GCN validation on the
separate validation graph. Final test evaluation uses the separate test graph
only after selection. The no-fixed-cost low-rank option remains
`initialization="none"`. Keep the same configuration and output directory to
resume candidates and cached evaluations.
To extend a completed run, pass its fingerprinted directory as `resume_from`,
increase `steps`, retain old checkpoints, and append new penalty values after
the old penalty list. Existing candidate indices and optimizer states are
preserved; appended penalty candidates start fresh.
