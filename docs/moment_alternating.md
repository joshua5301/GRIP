# Fixed-setting optimizer comparison

`src.moment_alternating.compare_optimizer` compares joint Adam against alternating
Adam on V + lambda ||M|| with random U,V logits scaled by sqrt(rank). Both reuse
the same features, teacher probabilities, seed, precision and learning rate.
This is block coordinate optimization, not closed-form alternating least squares.

Alternating uses separate persistent Adam states, at most 50 updates per block,
and switches earlier when the best objective's relative improvement over 10
updates is at most 1e-7. Only the active factor has gradients. Cell means are
recomputed and differentiated, never frozen. At block end restore both active
parameters and their Adam state at the lowest exact block objective, including
the block start. Accepted block objectives cannot increase. A fixed total update
budget is used; no global convergence claim is made.

History records actual J, best-so-far J, update count, active factor and synchronized
elapsed wall time. For joint Adam one update changes both factors; for alternating
it changes one. Compare equal updates and common elapsed time separately. Timing
includes block snapshots/rollbacks and per-update objective diagnostics, excludes
initial feature preparation and factor creation, and is not directly comparable
to the production partition_seconds field. No student validation or test is used.
Short Colab warmups and alternating execution order across seeds reduce timing bias.
