# Arxiv 0.05% width screening

`src.arxiv_width_sweep.run_arxiv_width_sweep` uses 90 cells and runs on a
Colab CUDA GPU. It fixes feature k-means++ initialization, 5% mixing, ReLU kernel
teacher, RMS-normalized S²X, mass-weighted inner CE, Newton-first exact solves,
and warm-start implicit differentiation. Features and labels remain joint means.

If the old selected parameters are available, pass
`baseline=dict(T=..., penalty=..., rank=..., lr=...)` and `gammas=[old_gamma]`.
Otherwise a six-point low-rank grid chooses a provisional baseline using
condensation seed zero, two search student seeds, and 200 assignment steps.
This fallback is a new baseline, not a reconstruction of the recorded 67.96%.
The teacher gamma grid is selected by teacher validation accuracy first.

The MLP grid is hidden width 128/512/1024 crossed with LR 0.003/0.01.
Temperature, rank and inner penalty are shared with the provisional baseline.
Screening uses 500-epoch students evaluated every 25 epochs. The best nonzero
checkpoint per candidate competes for promotion; retain two distinct widths,
with their screen-selected learning rates. Always retain the low-rank baseline.
Early screening can eliminate a width or learning rate that improves later.
This is a staged finite grid, not exhaustive full-budget grid search.

Promoted candidates receive 1000 assignment steps on condensation seeds 0/1/2.
Checkpoint selection uses full 1000-epoch students, evaluation every 10 epochs,
and the nine condensation/student seed pairs. Each family chooses one common
candidate and step by mean validation accuracy, preferring earlier steps on ties.
Only then do final student seeds 100..109 evaluate each family's step zero and
selected step. Condensation seeds participate in selection and are not held out.
Separate student variability from the sample standard deviation across three
condensation-seed means. The MLP family gets more trials than the baseline.

GCN hidden=256, dropout=0.31881090213944857, LR=0.01, weight decay=0.0005.
Synthetic adjacency is identity. Default GCN CE is uniform; `student_loss="mass"` starts a separate run
matching the historical Arxiv family runner.
The historical result's actual selected settings still require its saved artifacts.

With default settings, screening costs 12×200=2400 assignment updates. Extending
three candidates on seed zero costs 3×800=2400, and their two other seeds cost
3×2×1000=6000: total 10800. With a supplied fixed baseline, total is 9800.
The full grid across all three seeds would cost 36000 updates. GCN evaluations
and the teacher add substantial time, so these counts are not wall-clock estimates.

Saved outputs include config, teacher selection, initial equivalence checks,
candidate grid, screen scores, promotions, full validation/outer-CE curves,
selected checkpoints, per-student final results and summary. Optimizer logs also
record convergence, cell masses and timing. Identical arguments/output paths
resume checkpoint states and reuse evaluations. Optimizer resume files are saved
at checkpoints, so an interrupted interval is repeated. Partial GCN fits restart.
Source changes require a version bump if they change algorithm behavior.

`src.arxiv_loss_replay.run_arxiv_loss_replay(source_root)` freezes the uniform-CE
selected candidates/checkpoints and trains only the final GCNs with mass-weighted
CE. It reuses teacher logits, saved representatives and final seeds, verifies the
graph/split digest, and pairs each new score with its uniform-CE reference. No
condensation or checkpoint search runs. Default cost is 60 GCN fits, each with
1000 epochs and validation-selected student epoch. Outputs/cache are isolated
under `source_root/mass_replay`; identical reruns reuse completed student fits.
This is a loss ablation on uniform-selected condensates, not a separately tuned
mass-CE sweep. Test results do not change either family's candidate/checkpoint.

`src.arxiv_transfer_plot.plot_arxiv_transfer(source_root)` evaluates the saved
inner linear heads on validation nodes using the saved normalized S²X inputs.
It compares these trajectories with cached full-budget uniform-CE GCN search
validation scores on original X/adjacency. GCN student seeds are averaged within
each condensation seed first. Bands are sample SD across condensation seeds,
not confidence intervals. A second plot pairs outer teacher CE with GCN validation.
No new fitting or test scoring runs. CSV, PNG and PDF outputs are saved under
`source_root/transfer_diagnostic`. The classifiers and evaluation representations
differ, so their accuracy gap is a diagnostic rather than a controlled causal test.

Local verification is syntax/static checks only. Training and numerical tests
belong on Colab; local training remains disabled.
