# NTK risk partition with identity-graph reconstruction

`src.ntk_risk.run_ntk_risk` replaces only the original global risk partition's
feature space with the analytic two-layer GCN NTK. The original S²X kernel
teacher, cell-average soft labels and student training protocol are retained.
There is no KL term. The synthetic adjacency is fixed to identity in both
reconstruction and student training; validation/test use the original graph.

Let S be the original normalized adjacency including self loops and U=SX/√d.
The idealized scalar network is S√2 ReLU(SXW/√d)a/√h, with iid standard normal
weights and both W and a trainable, without biases or dropout. Its infinite-width
NTK is K=S ψ(U,U) Sᵀ, where, for angle θ between u and v,

    ψ(u,v)=||u||||v|| [sin θ + 2(π−θ)cos θ]/π.

The full PSD eigenfeature factor Φ satisfies ΦΦᵀ=K. The existing risk solver
centers and RMS-normalizes this feature space, then minimizes its original
global variance–feature/label moment objective B²V/4+2B||E||F. No rank truncation
is applied. The reconstruction below uses the unnormalized original kernel.

For each resulting cell Cj, the synthetic input is restricted to

    x̃j = Σ(i∈Cj) αji xi,  αji≥0,  Σi αji=1.

The original graph and identity graph use one common NTK. With Pij=1[i∈Cj]/nj,

    Ksynthetic,original = ψ(X̃/√d,U) Sᵀ,
    ej = 2||x̃j||²/d − 2[Ksynthetic,original P]jj + [PᵀKP]jj.

Adam optimizes Σj(nj/N)ej / mean(diag K) for 1000 updates by default. Softmax
cell weights start uniform; the lowest reconstruction loss iterate is returned.
`reconstruction_step` reports that iterate, not convergence. Every iteration's
current and best error are saved. Endpoint angles use a 1e−10 numerical
tolerance and finite subgradients; no reconstruction loss is clipped to zero.
This optimizes the feature of the mixed input, not a mixture of kernel features.
The target label remains the arithmetic mean of original teacher labels in the
cell, independently of reconstruction weights.

`raw_mean`, `raw_convex`, and optional `s2x_mean` share the same cached partitions
for identical teacher/B settings. Each mode selects its own best grid setting by
mean validation accuracy across search seeds. Test labels never enter tuning or
reconstruction; final student checkpoints are selected on validation and then
evaluated on test. `final.csv` records every final student seed. Cached artifacts
are separated by protocol, Git revision, dataset contents and splits.

The squared error is an RKHS mean reconstruction error. For a linear RKHS head
of norm at most B, replacing ideal centroids adds at most
√2 B Σj πj√ej ≤ √2 B√(Σjπj ej) to the mass-weighted soft CE discrepancy, for
fixed cell-average labels. If the partition features are RMS-normalized by s,
use ej/s² in this bound. This does not certify finite-width GCN training or its
generalization. The actual student retains the existing GCN biases, dropout and
Adam protocol; those are not exactly the idealized analytic NTK model. Uniform
student CE is available for comparison, but is not the mass-weighted risk in
this derivation. The old global moment proof applies to a linear kernel head,
not to an arbitrary nonlinear GNN merely because its input is a kernel feature.

This full N×N implementation is limited to Cora and CiteSeer. It is not an arxiv
implementation. Local training/tests are intentionally not run; the accompanying
Colab cell runs the kernel, cross-graph and convex reconstruction tests first.

## NTK versus trained GCN representation

`run_representation_risk` defaults to `representations=('ntk', 'gcn_teacher')`,
`modes=('raw_convex',)` and `loss_weighting='mass'`. Each representation receives
the same grid and student seeds, and independently selects settings on mean
search validation. All requested densities share one representation GCN per
dataset. The existing S²X kernel teacher supplies the same soft labels to both
branches for each gamma/T/basis setting. The representation GCN never supplies
the synthetic labels.

The representation teacher is a two-layer GCN, trained on original training
labels with full-graph propagation. Its checkpoint is selected by validation
accuracy, with no test evaluation. Defaults: hidden=256, dropout=.5, lr=.01,
weight_decay=.0005, epochs=1000, eval_every=10, seed=0. These fixed settings are
configurable via `representation_teacher`; they are not part of the student
grid. The teacher state and validation trajectory are cached.

Its representation is h=S ReLU(SXW1+b1), including the second propagation before
the final linear classifier. This ensures logits=hW2+b2. On the synthetic
identity graph, h̃=ReLU(X̃W1+b1). Partitioning uses h and the original global risk
solver. Reconstruction uses the same simplex weights, Adam steps, mass weighting,
raw X support, and cell-mean labels as NTK, replacing kernel distance by
||h̃j−mean(i in Cj)hi||². The teacher is frozen throughout reconstruction and
dropout is disabled. Error is normalized by original mean squared feature norm,
analogous to mean kernel diagonal in the NTK branch.

Both branches normalize partition features using the existing solver. Their J
and normalized reconstruction errors nevertheless describe different geometries
and should not be interpreted as a common cross-method quality score. This is a
comparison of an untrained analytic kernel and a supervised representation,
not a claim of identical representation-training cost or supervision usage.
All final students retain the same two-layer GCN architecture, identity
synthetic graph, mass CE, original-graph evaluation, and validation checkpoint
selection. Comparison summaries are saved beside the two protocol directories.

## Propagated-feature control

`representations=('s2x',)` uses the original pipeline's H=S²X directly, without
forming any NTK or training a representation GCN. The label teacher, partition
solver and student protocol are unchanged. Under A=I the corresponding synthetic
representation is simply X̃, so `raw_convex` minimizes
Σj πj||X̃j−mean(i in Cj)Hi||² / mean(i)||Hi||², with X̃j restricted to raw X
members of the same cell. It does not propagate the raw synthetic features a
second time. The optional `s2x_mean` mode directly uses the H cell averages as
synthetic input, reproducing the original global risk representative rule.
Both modes share partitions at identical settings but select their best settings
independently on validation. The raw convex mode is the matched reconstruction
control for the NTK and GCN-representation experiments; `s2x_mean` is the original
method control and has no convex reconstruction step.

## Large datasets: two main methods

`run_main_risk` runs exactly `(s2x, s2x_mean)` and `(ntk, raw_convex)`, both with
mass CE, identical label-teacher grids and student evaluation settings. It
defaults to Nyström for the NTK branch; `nystrom` configures landmarks (512),
feature block size (2048), seed (0), and eigenvalue cutoff (1e-10 relative).
The landmark count is independent of the label teacher's `basis` parameter.
`partition_block_size` controls the original risk solver's assignment blocks.

For U=SX/√d and the original base tangent kernel ψ, select landmark rows L of U.
With ψ(L,L)=VΛVᵀ, retain positive eigenvalues above the relative cutoff and set
R=VΛ^(-1/2). Construct Z=ψ(U,L)R in blocks and Φ=S Z. The graph kernel is
approximated by ΦΦᵀ. An identity-graph raw input has feature
φ(I,x)=ψ(x/√d,L)R. Both sides use the same landmarks and basis; reconstruction
minimizes mass-weighted squared distance to the means of Φ. This approximates
the base tangent kernel before the final graph propagation, not a separately
fitted embedding of each graph. No N×N kernel or dense N×M membership is built
in this path. Dense feature storage scales as O(Nr), while building the landmark
factor uses O(r²) memory. Dataset loading and sparse graph storage remain separate
costs; increasing rank still increases feature construction and partition cost.

The result is explicitly marked `ntk_backend=nystrom`. Saved diagnostics include
rank and relative Frobenius/trace errors on a small base-kernel probe set. These
diagnose the approximation before final propagation; they are not certificates
of the full graph-kernel error or student accuracy. The risk objective now acts
in the approximate feature space, so approximation error is an additional gap
relative to the exact analytic NTK used in the small-graph experiments.

The existing data protocol is preserved: arxiv is transductive; Flickr and Reddit
use separate train/validation/test induced subgraphs. Only the training graph is
used for inductive label-teacher fitting, landmarks, partitioning and reconstruction.
Validation and test evaluation run on their respective original subgraphs.
The cache fingerprints include graph contents and permit absent evaluation masks
for these inductive datasets. Exact full NTK on a large dataset is rejected.
Label-teacher logits are cached across the two branches by data, teacher seed,
kernel/basis/gamma settings and code revision, so changing only the representation
does not refit that same teacher. Temperature is applied to these shared logits.
