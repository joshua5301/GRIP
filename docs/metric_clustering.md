# Global linear metric, CE bilevel condensation

`src.metric_clustering.run_metric_sweep` learns one shared full-width matrix L.
The effective matrix is sqrt(d) L / ||L||_F, initialized to identity. It defines
the squared Mahalanobis cost ||L(z_i-c_j)||². Normalization fixes trace(LᵀL)=d,
separating overall distance scale from assignment temperature tau. There is no
learned bias, node embedding, cell embedding, independent representative, or
assignment-residual parameter. The unconstrained matrix stored by Adam is a
parameterization of this normalized metric; it retains rotational redundancy.

For each outer evaluation, restart centers from the same saved hard cell means.
Repeat a configured finite number of differentiable soft Lloyd updates:

    P_ij = softmax_j(-||L(z_i-c_j)||²/tau)
    m_j = sum_i P_ij
    c_j = sum_i P_ij z_i / m_j
    s_j = sum_i P_ij q_i / m_j
    pi_j = m_j/N.

P minimizes the distance-plus-negative-entropy objective for fixed centers.
The weighted mean minimizes the fixed-assignment quadratic distance for any L,
including singular L. The final representative is the mean of the final P;
finite unrolling does not claim a converged assignment/center fixed point.
Gradients propagate through every center update and through metric normalization.
Node-cell blocks are recomputed in backward instead of storing all assignment
matrices. No stop-gradient is applied to moving centers.

The inner student is a regularized linear softmax CE head on RMS S²X,
including a regularized bias. The inner solver must meet its gradient tolerance.
The outer loss is full-node teacher-target CE, using the implicit head Hessian
solve and differentiating its moment derivative through the unrolled clustering.
The Hessian solve must also meet tolerance; solver failures are not silently
accepted. Only L is updated by the outer Adam optimizer. This first version
uses exact solves, rather than the previous tracking approximation.

The initial experiment is Arxiv ratio .005, 909 cells. It reuses the historical
teacher gamma, saved logits, RMS transform and GCN hyperparameters. T, tau,
metric learning rate and head penalty are exposed as grid axes. No KL term or
old risk B is optimized. Optional risk initialization reuses its saved hard
partition, without retuning it at each temperature. The default is k-means++.

Compare hard initialization (step -1), identity metric with soft Lloyd (step 0),
and learned metric checkpoints. They are distinct initial datasets: the former
has hard means while the latter depends on tau and the number of Lloyd steps.
All are eligible for GCN validation selection. When step -1 wins, tau, penalty
and metric learning rate are not identified by that result; at step 0, penalty
and metric learning rate are likewise inactive for constructing the dataset.

GCN evaluation uses S²X coordinates obtained by undoing the saved
RMS scaling, A=I on the condensed graph, two layers, and mass-weighted CE. Search
and final seeds are disjoint. Test labels are accessed only after selecting the
candidate/checkpoint by mean GCN validation. Initial and last test results belong
to the selected candidate and are not used to rank candidates.

The original graph/split and teacher hashes are verified. Normalized node
features are saved once in inputs.pt and loaded exactly on resume, avoiding
strict hashes of recomputed GPU propagation. Resume state includes the matrix,
Adam state, warm-start head and next step after the saved update. Restarting
repeats only work after the last checkpoint. Code-versioned configurations and
cached per-seed evaluations isolate experiments; do not run the identical
configuration in concurrent sessions.

Colab tests compare streamed unrolled gradients against dense autograd, check
mass/feature/label preservation and metric scale invariance, compare the complete
bilevel derivative against finite differences with refitted students, and
exercise optimizer resumption. No local numerical experiments are run.
