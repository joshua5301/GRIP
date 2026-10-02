# Historical variance-moment grid

`src.variance_moment_sweep.run_risk_sweep` runs one density per session. Select
`dataset="cora"` with ratio 0.013 (35 representatives), 0.026 (70), or 0.052 (140),
or `dataset="citeseer"` with ratio 0.009 (30), 0.018 (60), or 0.036 (120).
`run_cora_risk_sweep` remains a compatible alias with Cora as the default.
The runner restores the original
partitioner and kernel teacher in memory from pinned commit
`12ceec5810e2c709f98f8aa518d401cd04390168`. A normal full Git clone is required.
The reference sources are copied into each result directory.

The grid is gamma `[0.001, 0.01, 0.1, 1]`, T `[0.2, 0.5, 1, 2]`, and B
`[0.3, 1, 3, 10, 30]` for Cora or `[0.03, 0.1, 0.3, 1, 3]` for Citeseer:
80 settings per density. Teacher features use ReLU for Cora and erf for Citeseer,
3000 basis nodes and teacher seed zero. Gamma is selected jointly by condensed GCN
validation, not by teacher accuracy. The teacher validation table is diagnostic.

The objective is the historical `B**2/4 * V + 2*B * ||M||_F`, using global
cross-moment discrepancy. Features are centered and RMS-normalized S²X inside
the partitioner; output representatives are restored to the original feature
scale. Native initialization samples distances in weighted features and teacher
labels; this is not feature-only k-means. B and T therefore affect initialization
as well as refinement. Condensation seeds also control node traversal order.
The solver stops when it accepts no moves or reaches 100 sweeps. Nonconvergence
is reported, not treated as successful convergence. No checkpoint is selected
by validation within a condensation run.

Every candidate uses condensation seeds 0, 1, 2 crossed with student seeds 0, 1, 2.
One shared hyperparameter tuple is selected by mean validation over all nine
pairs, with candidate order breaking ties. This is 240 condensations and 720
search student fits per density. Only the selected tuple receives final test
evaluation, using student seeds 100 through 109 for each condensation seed.
Students are two-layer GCNs with uniform soft CE, identity condensed adjacency,
hidden width 256, fixed dropout 0.9 for Cora or 0.5 for Citeseer, LR 0.01,
weight decay 0.0005, 1000 epochs and
validation every 10 epochs. The existing evaluator resets Adam at half the
training budget with LR multiplied by 0.1, matching the historical schedule.

`summary.csv` reports mean validation/test over condensation seeds and sample
standard deviations of those seed means. `by_seed.csv` separately reports the
student-seed variability within each condensation. The 30 final students are
not 30 independent condensations; condensation seeds participated in selection.

Use `/content/drive/MyDrive/GRIP_results` for `output_dir`. Configuration, graph
content, reference sources, relevant runner/evaluation code and PyTorch version
determine the run fingerprint. Propagated features are cached and reused within
that run, avoiding digest changes from recomputing sparse propagation. All
teacher, partition and student artifacts live below the run directory.
Re-running an identical cell reuses complete teachers, condensations and student
fits. An interrupted individual condensation or student fit restarts; completed
ones are skipped. Changing configuration or relevant code creates a separate run.
Do not let two sessions write to the same density/run concurrently.

Tests are in `tests/test_variance_moment_sweep.py`, intended for Colab. No local
training or numerical smoke tests are required.

## Flickr, Reddit and Arxiv

The same `run_risk_sweep` supports `dataset="flickr"`, `"reddit"` and `"arxiv"`.
Use three sessions, each selecting one column of this table, and run its three
dataset/density pairs sequentially:

| Dataset | Session 0 | Session 1 | Session 2 |
| --- | --- | --- | --- |
| Flickr | 0.001 / 44 nodes | 0.005 / 223 nodes | 0.01 / 446 nodes |
| Reddit | 0.0005 / 77 nodes | 0.001 / 153 nodes | 0.002 / 307 nodes |
| Arxiv | 0.0005 / 90 nodes | 0.0025 / 454 nodes | 0.005 / 909 nodes |

Each uses gamma `[0.001, 0.01, 0.1, 1]`, T `[0.2, 0.5, 1, 2]`, B
`[0.1, 0.3, 1, 3, 10]`, and fixed GCN dropout 0.5. The teacher kernel is ReLU
for Flickr/Arxiv and erf for Reddit. The other settings and selection protocol
are unchanged. Per session this is 720 condensations, 2160 search GCN fits and
90 final GCN fits. All 80 candidates use all nine search seed pairs; no pruning
or teacher-accuracy preselection is applied.

Flickr and Reddit use the existing inductive data loader: only their training
graphs supply condensation features, teacher fitting labels and kernel anchors.
Validation features propagate on the separate validation graph and use the same
training-anchor kernel mapping. Teacher caches store training and validation
logits. Student validation and final testing run on their respective separate
graphs; search evaluation receives no test split. Arxiv is transductive and
retains its original train/validation/test masks on the shared graph.

Graph fingerprints include each separate graph and mask. Use a distinct dataset
subdirectory under the results root; session aggregate CSVs and temporary folders
must also have different session names. As before, changed runner code changes
the run fingerprint; completed files in older run folders are retained.
