# ANIL representation study on Cora

`src.anil_representation.run_anil_representation` compares supervised CE and
ANIL using the same two-layer GCN initialization and outer optimizer settings.
The representation is S ReLU(S X W1 + b1), including the final propagation.
Dropout, when enabled, is applied before that final propagation. Default dropout
is zero to start with a deterministic encoder map.

Each ANIL episode samples class-balanced support nodes from the official train
mask; remaining train nodes form the query set. Only the final linear weight
and bias adapt in the inner loop. Functional SGD with create_graph=True retains
the exact unrolled gradient through all inner updates. The outer Adam update
learns both the encoder and the shared head initialization. The inner adaptation
never mutates the shared parameters. This is same-graph, same-class episodic
learning, not an evaluation of transfer to unseen tasks or classes.

Cora uses the transductive graph and features, including unlabeled validation and
test nodes, for message passing. Their labels never enter either training loss.
Validation selects encoder checkpoints by fitting a fresh head to a fixed train
subset. Test is evaluated only after selecting the encoder. The stored meta-head
is discarded for the reported frozen-encoder probes. Probe support sets and new
head initializations are paired between methods; fixed-length full-batch SGD
uses uniform CE. Reported deviations combine support sampling and head seeds,
not independent encoder training seeds.

The supervised control uses all train labels per outer update. Equal update
counts do not imply equal compute or identical label exposure. Training CSVs
include wall-clock time, checkpoint validation accuracy and CE. Both methods
save the selected state and full-node features for subsequent condensation.
This experiment evaluates representations, not condensed-graph GCN accuracy.

The run configuration, revision, data and splits identify the cache. An
interrupted method with an incomplete checkpoint requires restarting that method;
completed methods and final probes are reused. Local verification is restricted
to syntax and diff checks; numerical tests must run in Colab.
