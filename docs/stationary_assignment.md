# Fixed reference head, learned assignments

`src.stationary_assignment.run_stationary_assignment` uses the saved Arxiv
909-cell CE-bilevel reference to preserve its teacher, label temperature, RMS S²X,
CE penalty, smoothing, GCN settings and optional risk initialization. K-means++
initialization is also supported. Labels and cell mass follow the current dense
soft assignment; the condensed graph is identity and GCN CE is mass weighted.

First fit a regularized linear CE head on all original nodes and their fixed
teacher targets. This is the reference student, not a new ground-truth teacher.
Freeze its parameters, including its regularized bias. Optimize assignment logits
to minimize the squared Frobenius norm of the condensed regularized CE gradient
evaluated at this fixed head. The loss is divided by its initial value solely
for optimizer scaling; absolute gradient norm and theoretical bounds are logged.

With x_j=[c_j,1], the gradient is

    g(P) = sum_j pi_j (softmax(W_ref x_j) - s_j) x_j^T + lambda W_ref.

The implementation retains target row sums in the general gradient formula to
handle numerical soft-label normalization. It differentiates this explicit
expression through cell moments and soft assignments. No inner learner or
implicit Hessian solve runs inside assignment optimization. The reference fit
occurs once per session and is shared by both initializations. Optional final
diagnostics refit a condensed head after assignment optimization, never as part
of the optimization gradient.

Strong convexity yields ||W*_P-W_ref|| <= ||g(P)||/lambda. With an approximate
reference and full-data residual g_R, the distance between exact optimal heads
is bounded by (||g(P)||+||g_R||)/lambda. These are parameter-distance bounds for
the same regularized linear student, not guarantees of accuracy or GCN transfer.
Small lambda can make them loose. Actual diagnostic head distances use a
numerically fitted condensed head; its solve must pass the gradient tolerance.

GCN validation over separate search seeds selects among saved checkpoints,
including step zero. Final seeds evaluate the initial, selected and last states.
Ground-truth test labels are only used after selection. The full-node teacher
targets and graph features make this transductive, as in the prior experiments.
Each initialization has independent files; simultaneous sessions may run different
initializations but must not write to the same initialization directory. The
optimizer resumes from atomic saved states at checkpoints under the same config.

No local numerical tests were run. Colab tests compare the explicit gradient and
assignment derivative with autograd, check the strong-convexity distance bound on
a small problem, and exercise interrupted optimizer resumption with a fixed head.
