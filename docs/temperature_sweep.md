# Separate fixed inner and outer temperature grid

run_temperature_sweep uses all Cartesian combinations of the supplied temperatures.
Neither temperature is learned. Teacher logits, feature initialization, low-rank
factor initialization, rank, penalty and assignment learning rate come from the
selected source configuration. Only the two label temperatures vary.

The optimizer receives inner labels as q and outer labels as outer_targets. The
outer-target digest is part of resume validation. Equal temperatures recover the
shared-target objective. Each candidate starts independently and keeps its own
initial-outer-CE normalization, matching the existing optimizer convention.

Checkpoint and hyperparameter selection use the mean GCN validation accuracy over
search seeds, never outer CE (whose target entropy varies with outer temperature)
or test accuracy. Final evaluation uses separate seeds, mass CE, two-layer GCN and
identity synthetic adjacency. Report the validation-selected global best, best
shared-temperature setting, and source-temperature setting, selecting checkpoints
independently within each restriction. No test scores are computed for other grid
candidates. These are diagnostics on a fixed validation split, not independent
confirmation of tuning gains.

Default solver settings are Newton-first and implicit warm starts, with grouped
PCG and probability caching disabled following the timing result. Each candidate
resumes from its own checkpoint; evaluation seeds are cached. Source files remain
unchanged. Separate sessions use shard=0,1,2 and shards=3 with the same output path
and grid. Only the last-numbered shard aggregates and evaluates the final choices.
If it finishes before other sessions, rerun that shard after all sessions finish.
Do not run the same shard concurrently. Outputs are suffixed by shard to avoid
multiple writers for aggregated tables.
