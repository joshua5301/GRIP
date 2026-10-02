# Historical variance-moment grid

`src.variance_moment_sweep.run_cora_risk_sweep` runs one Cora density per session:
0.013 (35 representatives), 0.026 (70), or 0.052 (140). It restores the original
partitioner and kernel teacher in memory from pinned commit
`12ceec5810e2c709f98f8aa518d401cd04390168`. A normal full Git clone is required.
The reference sources are copied into each result directory.

The grid is gamma `[0.001, 0.01, 0.1, 1]`, T `[0.2, 0.5, 1, 2]`, and B
`[0.3, 1, 3, 10, 30]`: 80 settings per density. ReLU teacher features use 3000
basis nodes with teacher seed zero. Gamma is selected jointly by condensed GCN
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
hidden width 256, dropout 0.9, LR 0.01, weight decay 0.0005, 1000 epochs and
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
