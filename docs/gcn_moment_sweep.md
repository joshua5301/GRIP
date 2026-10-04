# GCN teacher and variance-moment clustering

`src.gcn_moment_sweep.run_gcn_moment_sweep` trains the evaluation GCN on the
original training graph with hard training labels. Validation accuracy selects
the checkpoint; test metrics are computed after checkpoint selection. Its
fixed log probabilities supply temperature-scaled soft labels. This is a
single-seed Full GCN teacher, not a multi-seed Full accuracy average.

Clustering uses centered, scalar-RMS-normalized S²X, feature-only Var-Part,
and filtered-batch descent on `V + lambda * ||M||_F`. Neither labels nor the
moment are variance-normalized. Representatives restore the S²X scale.
All candidates start with the same deterministic feature partition. Students
are fresh GCNs trained with uniform CE and identity condensed adjacency.

Teacher and student settings share the architecture, dropout, Adam learning
rate, weight decay, and halfway optimizer reset at one-tenth learning rate.
Inductive datasets retain separate training, validation, and test graphs.
Only T and lambda are swept; there is no kernel teacher or gamma grid.

Completed artifacts are cached under a configuration/data/code fingerprint.
Interrupted individual fits restart. Final student seeds are disjoint from
validation search seeds. Run `tests/test_gcn_moment_sweep.py` in Colab to check
checkpoint restoration and test-only-after-selection behavior.

Optional `teacher_dropouts` and `teacher_weight_decays` grids select a teacher
by validation accuracy, then validation CE, then grid order. Teacher seed and
learning rate stay fixed; student settings are unchanged. No candidate test
metrics are computed during selection. The selected teacher alone receives a
test evaluation. `teacher_grid.csv` and `selected_teacher.json` record selection.

Pass this stage-one run to `distance_finetune.run_distance_finetune` to refine
assignments with fixed moment-linearization costs. It uses exactly the saved
S²X, centered scalar RMS normalization, selected temperature, and teacher.
Costs are row-centered and divided by the median positive top-two cost gap.
The fixed cost is stored as exact rectangular factors rather than an N-by-K
matrix. Features and mean labels are decoded from soft assignment moments;
features are restored to the original S²X scale for GCN evaluation.
Hard stage one, soft step zero, and validation-selected learned results are
reported separately. `inner_tol` and `cg_rtol` configure solver tolerances.
The second stage minimizes teacher CE through the fitted linear head, not
the first-stage variance-moment objective.
