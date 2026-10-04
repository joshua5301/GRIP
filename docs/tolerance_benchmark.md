# Solver tolerance benchmark

`src.tolerance_benchmark.run_tolerance_benchmark` loads a distance-finetuning run's
stage-one partition, features and teacher. It compares strict `(1e-7, 1e-6)`,
relaxed `(1e-6, 1e-4)` and loose `(1e-5, 1e-3)` inner-gradient and implicit-CG
tolerances. All trials keep initialization, objective, learning rate and step
budget fixed, with warm starts for both solvers. Alternate repeats reverse order.

The GPU-synchronized optimization timer excludes data preparation, distance/SVD
initialization, final artifact serialization and student evaluation. It includes
optimization logging and checkpoint construction. Per-step averages exclude step
zero and the final evaluation. Run on an otherwise idle GPU.

Each final checkpoint is verified with a strict inner solve before reporting its
outer CE. GCN validation uses uniform CE and the reference's search seeds; test
accuracy is not evaluated. The comparison measures a fixed step budget, without
selecting checkpoints by validation. Files remain inside the supplied output
directory; completed timing trials are reused on rerun.

An optional `tolerances` mapping overrides the default levels, for example
`{"very_loose": (1e-4, 1e-2), "extreme": (1e-3, 1e-1)}`. Strict final
verification remains unchanged regardless of the optimization tolerances.
