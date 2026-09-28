# Dense, low-rank and feature-conditioned assignment comparison

`src.assignment_family_sweep.run_family_sweep` runs a full finite grid on Arxiv.
The requested 0.05% is ratio=0.0005: 90 cells in the repository's benchmark
budget table, not the previous ratio=.005 (909 cells).

First select a single kernel teacher gamma by highest ground-truth validation
accuracy. The existing teacher fitter uses training labels only; basis features
use the transductive graph. No test labels participate in teacher selection.
The selected logits and gamma are reused by every condensation candidate. T is
the subsequent soft-label temperature and can still be swept independently.

All methods share the same k-means++ initialization on S²X, RMS feature
normalization, and 5% uniform mixing. Compare these parameterizations:

| Method | Trainable assignment score | Number of parameters |
| --- | --- | --- |
| dense | S, initialized to S0 | NM |
| low_rank | S0 + UVᵀ/sqrt(r) | r(N+M) |
| mlp1 | S0 + (zW)Vᵀ/sqrt(r) | r(d+M) |
| mlp2 | S0 + (ReLU(zW1+b1)W2+b2)Vᵀ/sqrt(r) | dh+h+hr+r+Mr |

The one-layer encoder is a linear map without bias or activation, matching the
existing implementation. Its output width is r; there is no separate hidden
width. The two-layer encoder has hidden width h and output width r, both swept.
Each node and cell factor uses r dimensions. V is trainable in all factorized
variants. Node factors or encoder output layers start at zero, preserving S0.
Low rank refers to the learned correction, not necessarily the total score or P.

These are the existing CE-bilevel algorithms, not learned distances or Lloyd
updates. Softmax rows sum to one, columns are free, features/labels are joint
weighted means and student CE is mass weighted. Initial moments are compared
numerically across all families at each T before accepting their evaluations.

Every grid point receives the same step budget and GCN search seeds. No pruning
is performed. Families with more rank/width variants have more search trials;
this is a tuned-family comparison, not an equal-total-search-budget benchmark.
Each family independently selects its setting and checkpoint, including step
zero, by mean two-layer GCN validation accuracy. Final seeds are disjoint. Test
is evaluated only after all searches finish, at the selected and matched initial
states for each family. A selected step zero provides no evidence favoring its
rank, width, penalty or assignment learning rate. Initializations are shared,
but this experiment does not average over multiple condensation seeds.

The default runner uses the existing exact CE inner and adjoint solves. Tracking
can be supplied explicitly via the solver dictionary, uniformly for all families.
All other existing solver arguments are preserved. Normalized features, source
S²X and assignments are saved once; resume does not require bitwise agreement
with newly recomputed GPU propagation. Optimizer resumes use the existing saved
parameter and Adam state. Completed candidates and per-seed GCN evaluations are
cached. Rerun the identical cell to resume. Different method subsets have
separate protocol directories; avoid concurrently writing the same configuration.

The returned table contains initial and selected performance for each family.
`candidates.csv`, `search.csv`, `selected_<method>.json`, `initial_checks.csv`,
`teacher_grid.csv` and `final_students.csv` support capacity, boundary, and paired
student-seed analyses. Tests exercise initial equality, parameter counts,
candidate enumeration, and validation-only selection. Numerical tests are for
Colab; local verification is limited to static parsing and diff checks.
