import statistics
import time

import torch

from src.head import head_objective
from src.moments import augmented
from src.soft_ce_partition import solve_inner, solve_inner_newton_first


class PairedInnerSolver:
    def __init__(self, repeats=2, newton_steps=8):
        self.repeats, self.newton_steps, self.calls = repeats, newton_steps, 0

    def __call__(
        self, centers, labels, mass, penalty, initial=None, max_iter=2000, grad_tol=1e-7, cg_max_iter=512
    ):
        def clock():
            if centers.is_cuda:
                torch.cuda.synchronize(centers.device)
            return time.perf_counter()

        times = dict(lbfgs=[], newton=[])
        fitted = {}
        for repeat in range(self.repeats):
            order = ("lbfgs", "newton") if (self.calls + repeat) % 2 == 0 else ("newton", "lbfgs")
            for mode in order:
                started = clock()
                if mode == "lbfgs":
                    result = solve_inner(
                        centers, labels, mass, penalty, initial, max_iter, grad_tol, cg_max_iter=cg_max_iter
                    )
                else:
                    result = solve_inner_newton_first(
                        centers,
                        labels,
                        mass,
                        penalty,
                        initial,
                        max_iter,
                        grad_tol,
                        cg_max_iter=cg_max_iter,
                        newton_steps=self.newton_steps,
                    )
                times[mode].append(clock() - started)
                fitted[mode] = result
        left, right = fitted["lbfgs"]["theta"], fitted["newton"]["theta"]
        with torch.no_grad():
            x = augmented(centers)
            reference, proposal = x @ left.T, x @ right.T
            extra = dict(
                benchmark_first_mode="lbfgs" if self.calls % 2 == 0 else "newton",
                benchmark_theta_relative_difference=float(
                    (left - right).norm() / left.norm().clamp_min(1e-30)
                ),
                benchmark_objective_difference=float(
                    (
                        head_objective(x, labels, mass, left, penalty)
                        - head_objective(x, labels, mass, right, penalty)
                    ).abs()
                ),
                benchmark_probability_difference=float(
                    (reference.softmax(1) - proposal.softmax(1)).abs().max()
                ),
                benchmark_newton_fallback=fitted["newton"]["inner_lbfgs_fallback"],
                benchmark_newton_steps=fitted["newton"]["inner_newton_steps"],
                benchmark_newton_cg_iterations=fitted["newton"]["inner_newton_cg_iterations"],
            )
        for mode in ("lbfgs", "newton"):
            extra[f"benchmark_{mode}_seconds"] = statistics.median(times[mode])
            for key in ("inner_grad_max", "inner_converged", "inner_iterations", "inner_polish_steps"):
                extra[f"benchmark_{mode}_{key}"] = fitted[mode][key]
        self.calls += 1
        return dict(fitted["lbfgs"], **extra)
