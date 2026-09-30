# Learnable clustering for graph condensation

Learn a soft assignment from graph nodes to condensed cells. Features and labels
are assignment-weighted means; the condensed graph uses identity adjacency.

The current protocol uses S²X features, a kernel teacher, cell-mass-weighted inner
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
condensation seeds × three student seeds). Final evaluation uses ten student seeds
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
