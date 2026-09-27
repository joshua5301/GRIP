# NTK geometry transformations

The study uses a common analytic GCN base kernel with a fixed Nyström map and
landmarks. This is not an empirical Jacobian NTK. All conditions use the same
approximation, labels and student protocol. Transforms currently require the
explicit Nyström backend, even for Cora; they do not silently approximate the
exact-kernel reconstruction formula.

`rms` centers features and scales their mean squared vector norm to one. `l2`
first centers, then divides each vector by its norm. `whiten` applies the
symmetric covariance transform (C + epsilon I)^(-power/2), with epsilon equal
to ridge times the largest covariance eigenvalue. Negative numerical eigenvalues
are clipped to zero. A final centering and RMS normalization is applied to all
conditions to prevent global scale from masquerading as a geometry change.

Statistics are fitted once on the condensation graph without labels. For Cora
this is the full transductive graph. For inductive datasets it is the training
graph. The identical frozen transform is applied to identity-graph synthetic
features during raw-X convex reconstruction. Targets are means of transformed
node features, not transforms of the means. L2 and whitening change the kernel
and its norm constraint; they are not exact reparameterizations preserving the
original NTK norm ball. No guarantee of a smaller head norm or risk bound follows.

`run_ntk_risk(feature_transform=...)` runs each condition, retuning B alongside
teacher label parameters and student hyperparameters on validation. Mass CE,
cell-mean labels, raw within-cell convex inputs and A=I are retained. Transformed
features and statistics are saved in each dataset folder. The extra regularized
linear probe uses all 140 Cora train labels, fits in float64 with SGD and L-BFGS,
and selects penalty/lr by validation CE only. Test is evaluated after selection.
The fitted head norm is a diagnostic, not a verified uniform bound on students.

The probe measures the representation's label prediction ability. The separate
condensation experiment measures fresh two-layer GCN students. These accuracies
are different quantities. A stronger probe need not imply a stronger condensed
dataset. Existing NTK approximation and finite-student mismatch remain.

Tests cover global rescaling invariance, zero whitening power, frozen transform
statistics, finite reconstruction gradients and a realizable linear-map target.
Run numerical tests in Colab; local checks are syntax-only.
