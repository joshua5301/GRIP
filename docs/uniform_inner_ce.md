# Uniform inner and evaluation CE

`inner_loss_weighting='uniform'` replaces the CE weights of condensed cells with
1/M in inner fitting, Hessian operators and implicit mixed derivatives. It does
not balance the assignment columns or alter centroid/label barycenters. Geometric
cell mass continues to divide both feature and label sums, so its derivatives
through these ratios remain present. Only the multiplicative CE weight is fixed.
The penalty convention is unchanged; both CE weight schemes sum to one.

`compare_loss_weighting` compares mass_CE with uniform_CE from the selected source
configuration and identical initial representatives. Each uses matching weights in
its linear inner model and two-layer GCN evaluation. The outer objective stays a
uniform average across original nodes. Teachers, both temperatures, rank, penalty,
assignment learning rate, seed and checkpoint budgets are held fixed. The runner
uses Newton-first and implicit warm starts without grouped PCG or probability
caching. GCN checkpoint selection uses search validation seeds, and final scores
use separate paired seeds. The initial GCN scores can differ because the training
weights differ, despite identical synthetic features and labels.

This is a controlled change of loss weighting, not a separately tuned uniform-CE
baseline. A weaker result does not establish that uniform CE cannot match mass CE
after retuning. Mass weighting is part of the method and is not automatically an
unfair evaluation, but reporting standard uniform-CE results removes dependence on
custom evaluation weights. Historical resume files retain their mass-CE behavior;
new uniform runs store the weighting and reject incompatible resume settings.

Numerical tests check the uniform implicit hypergradient against finite differences
and verify that initial moments are unchanged. Run them on Colab, not locally.
