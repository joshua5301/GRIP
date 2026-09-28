# Fixed-head curvature audit

Reuse saved stationary-assignment moments and the fixed reference head. Do not optimize assignments or retrain GCNs. Fit a diagnostic condensed SGC head at each saved step, with the same mass CE, regularization (including bias), and representation. GCN validation comes from the original search CSV; test results are not used.

Let g be the condensed regularized CE gradient at W_ref, H its Hessian there, W_P its independently converged optimum, and delta = W_ref - W_P. Measure the ordinary gradient norm, actual head distance, local Newton displacement H^{-1}g, its relative error and cosine against delta, and the local objective-gap approximation g^T H^{-1}g / 2. These Newton quantities are local approximations, not guarantees.

The exact identity is g(W_ref) - g(W_P) = integral_0^1 H(W_P + t delta) delta dt. Gauss-Legendre quadrature checks this identity, with half as many points as an integration-resolution check. Path directional curvature is delta^T (g(W_ref)-g(W_P)) / ||delta||^2, computed from endpoint gradients without quadrature error. Compare it with local curvature along delta and along g. This examines relevant directions, not the entire Hessian spectrum or its minimum eigenvalue.

Small path curvature relative to gradient-direction curvature supports directional sensitivity as an explanation. Accurate Newton displacement with weak ordinary residual/head-distance agreement supports a curvature-aware diagnostic; inaccurate Newton displacement means local linearization itself is insufficient. Neither outcome establishes GCN transfer or causality. Do not use these diagnostics to select by test accuracy.

Checkpoint JSON caches include moments/reference digests, penalty, and solver-option-specific output folders. Interrupted audits reuse completed checkpoints. The diagnostic fits and Hessian solves are additional work and are not part of the original optimization time.
