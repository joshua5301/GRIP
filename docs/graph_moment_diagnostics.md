# Paired graph-moment diagnostics

`src.graph_moment_diagnostics.run_graph_moment_diagnostics` reads a completed
graph-moment sweep. It does not rerun condensation, k-means, teacher fitting or
hyperparameter search.

## Matched-temperature students

The selected moments/best.json supplies T, dropout, lr and weight decay for both
initial and optimized graphs. Initial raw features and graph come from saved
artifacts. Both versions use exactly the optimized artifact's mean labels and
cell masses. Width, CE weighting, epochs and validation checkpoint selection
are inherited from the source protocol. Student seeds are paired (100..109 by
default). Per-seed validation/test deltas are optimized minus initial in percentage
points. Saved final results are reused only when their settings, runtime and seed
match; initial results selected at a different T are not reused.

## Risk transfer

Original matching probes are loaded from cache. Ten new probe seeds 6000..6009
are trained under the identical source probe protocol: K random full-graph nodes,
teacher T=1 soft targets, original graph, fixed epoch checkpoints. They never use
validation/test ground truth. At evaluation, both graphs use the selected T.

For every fixed checkpoint the experiment evaluates original full-node soft CE
and each synthetic graph's mass-weighted soft CE. It records signed and absolute
gaps, along with the same first/second moment errors as condensation. Risk metrics
remain mass weighted even if the source chose uniform downstream student CE;
student accuracy evaluation inherits that source choice explicitly.

Summaries separate epochs 0/50/200 rather than treating checkpoints from the same
seed as independent repeats. Positive gap_change means worse preservation; negative
means improvement. Matching probes are an in-sample reference; held-out probes use
independent seeds but the same training distribution. They are not the students
trained on synthetic graphs and do not establish a uniform risk guarantee.

## Artifacts and resumption

Source protocol and selected artifacts are fingerprinted along with the new seed
sets and evaluation identity. Outputs include students.csv, student_summary.csv,
student_changes.csv, probe_risks.csv, risk_summary.csv, risk_changes.csv and
diagnostics.png. Student and probe measurements resume individually. The held-out
probe bank is cached. A matching teacher is discovered inside the source output's
teacher directory; pass teacher_run for an externally saved teacher.

The plot shows paired student accuracy, CE gaps by probe epoch and cohort, and
held-out before/after scatter. The diagonal in the latter is equality, with lower
points indicating improvement. Error bars are seed standard deviations.

No local training or smoke tests are run. Colab verification:

    python -m pytest -q tests/test_graph_moment_diagnostics.py
