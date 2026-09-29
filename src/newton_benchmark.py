import json
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from src.assignment_family_sweep import family_options
from src.cmean_family_sweep import write_table
from src.condensation_diagnostics import save_json
from src.risk_experiment import _fingerprint
from src.soft_ce_partition import optimize_ce_assignment, solve_inner, solve_inner_newton_first
from src.soft_ridge_partition import augmented
from src.stationarity_risk import head_objective


class PairedInnerSolver:
    def __init__(self, repeats=2, newton_steps=8):
        self.repeats, self.newton_steps, self.calls = repeats, newton_steps, 0

    def __call__(self, centers, labels, mass, penalty, initial=None, max_iter=2000,
                 grad_tol=1e-7, cg_max_iter=512):
        def clock():
            if centers.is_cuda:
                torch.cuda.synchronize(centers.device)
            return time.perf_counter()

        times = dict(lbfgs=[], newton=[])
        fitted = {}
        for repeat in range(self.repeats):
            order = ('lbfgs', 'newton') if (self.calls + repeat) % 2 == 0 else ('newton', 'lbfgs')
            for mode in order:
                started = clock()
                if mode == 'lbfgs':
                    result = solve_inner(centers, labels, mass, penalty, initial, max_iter,
                                         grad_tol, cg_max_iter=cg_max_iter)
                else:
                    result = solve_inner_newton_first(centers, labels, mass, penalty, initial, max_iter,
                        grad_tol, cg_max_iter=cg_max_iter, newton_steps=self.newton_steps)
                times[mode].append(clock() - started)
                fitted[mode] = result
        left, right = fitted['lbfgs']['theta'], fitted['newton']['theta']
        with torch.no_grad():
            x = augmented(centers)
            reference, proposal = x @ left.T, x @ right.T
            extra = dict(
                benchmark_first_mode='lbfgs' if self.calls % 2 == 0 else 'newton',
                benchmark_theta_relative_difference=float((left - right).norm() / left.norm().clamp_min(1e-30)),
                benchmark_objective_difference=float((head_objective(x, labels, mass, left, penalty)
                                                     - head_objective(x, labels, mass, right, penalty)).abs()),
                benchmark_probability_difference=float((reference.softmax(1) - proposal.softmax(1)).abs().max()),
                benchmark_newton_fallback=fitted['newton']['inner_lbfgs_fallback'],
                benchmark_newton_steps=fitted['newton']['inner_newton_steps'],
                benchmark_newton_cg_iterations=fitted['newton']['inner_newton_cg_iterations'])
        for mode in ('lbfgs', 'newton'):
            extra[f'benchmark_{mode}_seconds'] = statistics.median(times[mode])
            for key in ('inner_grad_max', 'inner_converged', 'inner_iterations', 'inner_polish_steps'):
                extra[f'benchmark_{mode}_{key}'] = fitted[mode][key]
        self.calls += 1
        return dict(fitted['lbfgs'], **extra)


def benchmark_newton_first(previous_run, output_dir, windows=('original', 'extended'),
                           updates=50, repeats=2, warmup=5, newton_steps=8, device='cuda'):
    if updates <= warmup or warmup < 0 or repeats < 1 or newton_steps < 0 or not windows or set(windows) - {'original', 'extended'}:
        raise ValueError('Invalid timing configuration')
    source = Path(previous_run)
    prior = json.loads((source / 'config.json').read_text())
    if prior['solver'].get('solver_mode', 'exact') != 'exact':
        raise ValueError('Use an exact source experiment')
    choice = json.loads((source / 'selected_low_rank.json').read_text())
    candidate = source / f'candidate_{int(choice["candidate"]):04d}'
    params = json.loads((candidate / 'params.json').read_text())
    paths = {w: candidate / ('extended/resume.pt' if w == 'extended' else 'resume.pt') for w in windows}
    config = dict(source=str(source), prior=prior, params=params, windows=list(windows), updates=updates,
        repeats=repeats, warmup=warmup, newton_steps=newton_steps, implicit_warm_start=True, version=1,
        sources={k: dict(path=str(p), size=p.stat().st_size, modified=p.stat().st_mtime_ns) for k, p in paths.items()})
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / 'config.json')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(source / 'inputs.pt', map_location=device, weights_only=False)
    z, assignment = saved['z'], saved['assignment']
    del saved
    teacher = torch.load(source / 'teacher.pt', map_location=device, weights_only=False)
    q = (teacher['logits'] / params['T']).softmax(1).double()
    del teacher
    tables = []
    for window, path in paths.items():
        folder = root / window
        folder.mkdir(exist_ok=True)
        cache = folder / 'timings.csv'
        if cache.exists():
            tables.append(pd.read_csv(cache))
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)
        state['config'] = dict(state['config'], implicit_warm_start=True)
        options = dict(prior['solver'], implicit_warm_start=True)
        start = int(state['step'])
        solver = PairedInnerSolver(repeats, newton_steps)
        result = optimize_ce_assignment(z, q, assignment, penalty=params['penalty'], lr=params['assignment_lr'],
            steps=start + updates, mixing=prior['mixing'], factor_seed=prior['seed'],
            folder=folder, resume_state=state, save_assignment=False, save_resume=False,
            checkpoint_steps=(), inner_solver=solver, **family_options(params), **options)
        history = pd.DataFrame(result['history'])
        history = history[(history.step >= start) & (history.step < start + updates)].copy()
        history['window'] = window
        history['offset'] = history.step - start
        history['included'] = history.offset >= warmup
        write_table(history, cache)
        tables.append(history)
        del state, result, solver
    timings = pd.concat(tables, ignore_index=True)
    rows = []
    for window, group in timings[timings.included].groupby('window', sort=False):
        old, new = group.benchmark_lbfgs_seconds, group.benchmark_newton_seconds
        other = group[['assignment_seconds', 'outer_seconds', 'implicit_seconds', 'backward_seconds']].sum(axis=1)
        rows.append(dict(window=window, pairs=len(group), lbfgs_ms=1000 * old.mean(), newton_ms=1000 * new.mean(),
            inner_speedup=old.sum() / new.sum(), estimated_step_speedup=(other + old).sum() / (other + new).sum(),
            inner_share_percent=100 * old.sum() / (other + old).sum(),
            lbfgs_iterations=group.benchmark_lbfgs_inner_iterations.mean(),
            newton_steps=group.benchmark_newton_steps.mean(),
            newton_cg_iterations=group.benchmark_newton_cg_iterations.mean(),
            fallback_percent=100 * group.benchmark_newton_fallback.mean(),
            lbfgs_converged=bool(group.benchmark_lbfgs_inner_converged.all()),
            newton_converged=bool(group.benchmark_newton_inner_converged.all()),
            lbfgs_gradient_max=group.benchmark_lbfgs_inner_grad_max.max(),
            newton_gradient_max=group.benchmark_newton_inner_grad_max.max(),
            theta_difference_max=group.benchmark_theta_relative_difference.max(),
            objective_difference_max=group.benchmark_objective_difference.max(),
            probability_difference_max=group.benchmark_probability_difference.max()))
    summary = pd.DataFrame(rows)
    write_table(summary, root / 'summary.csv')
    write_table(timings, root / 'timings.csv')
    return summary, timings, root
