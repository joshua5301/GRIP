# Cora joint condensation-seed grid

Run one density per session: 0.013 (35 cells), 0.026 (70), and 0.052 (140).
Each candidate is a Cartesian combination of T, rank and inner penalty. Assignment
learning rate is fixed. Condensation seeds 0, 1 and 2 determine both feature k-means
and low-rank factors. Each candidate/checkpoint is scored using the mean GCN
validation accuracy over all three condensation seeds and three student seeds.
Choose one common hyperparameter tuple and checkpoint, not separate best settings
or steps per condensation seed. Missing or duplicate seed pairs fail aggregation.

The teacher uses a separate fixed seed and gamma is selected once per density run
by teacher validation accuracy, before condensation. Independent sessions use the
same data, teacher seed, kernel, basis and gamma grid, so they reproduce the same
teacher selection without sharing writable cache files. selected_teacher.json
records gamma and a logit digest to verify agreement between sessions. No teacher
variation is introduced across condensation seeds within a density.

Features are RMS-normalized S^2X for optimization, with the existing inverse scale
used for synthetic GCN inputs. Low-rank corrections start from k-means assignments.
Inner linear CE is cell-mass weighted; outer teacher CE is uniform over original
nodes; both targets use the same fixed T. Final GCN training has two layers,
identity synthetic adjacency and uniform CE. Newton-first and implicit warm start
are enabled; grouped PCG and probability caching remain disabled.

After validation selection, evaluate the chosen candidate at step zero and the
selected step for each condensation seed using student seeds 100 through 109.
Report per-condensation-seed means and student standard deviations, the overall
mean, and the sample standard deviation of the three condensation-seed means.
The 30 student fits are not 30 independent condensations. The three condensation
seeds also participate in hyperparameter selection and are not held-out seeds.

Output is separate per ratio/configuration. Every candidate/condensation seed has
its own resumable optimizer state and cached GCN evaluations. Rerunning the same
cell resumes interrupted work. Changing the grid or configuration creates a new
run. A rerun skips completed optimization and evaluation; it recomputes aggregation.
Only selected configurations receive final test evaluation. No local numerical
tests or training are needed; run the provided tests in Colab.
