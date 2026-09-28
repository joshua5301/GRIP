# Initialization versus assignment parameterization

`src.initialization_factorial.run_initialization_factorial` uses the saved
300-step Arxiv CE bilevel run (`arxiv_909_ce_bilevel_v1/ad076fd3fe04`) as its
reference. Required artifacts are its config and optimized initial moments, the
previous ridge/CE run's reference transform, and the source teacher logits and
variance-moment baseline partition. It refuses a graph, split, teacher or
partition mismatch rather than silently constructing a new reference.

The four cases are risk_dense, kmeans_dense, risk_mlp and kmeans_mlp. Both
initializations use the same saved teacher probabilities and temperature, RMS
transform, smoothing, positive CE penalty, assignment learning rate, student
hyperparameters and 300-step budget. K-means++ runs on S²X. Risk uses the exact
saved partition. Dense optimizes all node-cell logits; MLP uses feature inputs,
64 hidden units and a rank-16 assignment correction. Labels and masses change
with assignments in every case, and student CE is mass weighted throughout.

Every case uses converged inner/implicit solves, not tracking, to avoid that
additional confound. The current numerical solver includes residual-checked
fallbacks, so this is a reference reproduction attempt, not a promise of bitwise
identity with the historical code. Equal learning rates are a controlled setting,
not evidence that each parameterization has received its own optimal tuning.

GCN validation averaged over search seeds selects a checkpoint, including zero.
Separate final seeds evaluate initial, selected and fixed step-300 states. Test
does not select any setting. Fixed step 300 permits comparison with the reported
70.093822 historical result independently of the new selection protocol.
SGC validation/CE and per-student epoch curves are also saved. The matched initial
moment checks distinguish changes in parameterization from changes in its starting
representatives. These experiments condition on a single partition/encoder seed.

All cases can run in one session, or subsets can run in separate sessions. Each
case writes into its own directory under a shared protocol hash. Do not run the
same case concurrently. Adam/head state resumes from saved checkpoints; cached
GCN seed evaluations are reused. No local numerical experiments were run.
