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
from src.soft_ce_partition import optimize_ce_assignment, solve_head_system


class PairedHessianSolver:
    def __init__(self, initial=None, repeats=2):
        self.previous = initial
        self.repeats = repeats
        self.calls = 0

    def __call__(self, x, labels, mass, theta, penalty, rhs, **options):
        def clock():
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            return time.perf_counter()

        initial = self.previous.to(rhs) if self.previous is not None else None
        times = dict(cold=[], warm=[])
        solutions, diagnostics = {}, {}
        for repeat in range(self.repeats):
            order = ('cold', 'warm') if (self.calls + repeat) % 2 == 0 else ('warm', 'cold')
            for mode in order:
                started = clock()
                value, diagnostic = solve_head_system(
                    x, labels, mass, theta, penalty, rhs,
                    initial=initial if mode == 'warm' else None, **options)
                times[mode].append(clock() - started)
                solutions[mode], diagnostics[mode] = value, diagnostic
        self.previous = (solutions['warm'] if diagnostics['warm']['cg_converged'] else solutions['cold']).detach()
        self.calls += 1
        extra = dict(benchmark_has_previous=initial is not None,
                     benchmark_first_mode='cold' if (self.calls - 1) % 2 == 0 else 'warm',
                     benchmark_solution_relative_difference=float(
                         (solutions['cold'] - solutions['warm']).norm() / solutions['cold'].norm().clamp_min(1e-30)))
        for mode in ('cold', 'warm'):
            extra[f'benchmark_{mode}_seconds'] = statistics.median(times[mode])
            for key in ('cg_iterations', 'cg_relative_residual', 'cg_converged', 'hessian_solver'):
                extra[f'benchmark_{mode}_{key}'] = diagnostics[mode][key]
        return solutions['cold'], dict(diagnostics['cold'], **extra)


def benchmark_low_rank_hessian(previous_run, output_dir, windows=('original', 'extended'),
                               updates=50, repeats=2, warmup=5, device='cuda'):
    if updates <= warmup or warmup < 0 or repeats < 1 or not windows or set(windows) - {'original', 'extended'}:
        raise ValueError('Invalid timing windows, repeats or warmup')
    source = Path(previous_run)
    prior = json.loads((source / 'config.json').read_text())
    if prior['solver'].get('solver_mode', 'exact') != 'exact':
        raise ValueError('Use an exact-mode source experiment')
    choice = json.loads((source / 'selected_low_rank.json').read_text())
    candidate = source / f'candidate_{int(choice["candidate"]):04d}'
    params = json.loads((candidate / 'params.json').read_text())
    paths = {window: candidate / ('extended/resume.pt' if window == 'extended' else 'resume.pt') for window in windows}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    config = dict(source=str(source), prior=prior, candidate=params, windows=list(windows), updates=updates,
                  repeats=repeats, warmup=warmup, source_states={k: dict(path=str(v), size=v.stat().st_size,
                  modified=v.stat().st_mtime_ns) for k, v in paths.items()}, version=1)
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
        start = int(state['step'])
        solver = PairedHessianSolver(state.get('tracking_vector_before'), repeats)
        result = optimize_ce_assignment(
            z, q, assignment, penalty=params['penalty'], lr=params['assignment_lr'],
            steps=start + updates, mixing=prior['mixing'], factor_seed=prior['seed'],
            folder=folder, resume_state=state, save_assignment=False, save_resume=False,
            checkpoint_steps=(), implicit_solver=solver, **family_options(params), **prior['solver'])
        history = pd.DataFrame(result['history'])
        history = history[(history.step >= start) & (history.step < start + updates)].copy()
        history['window'] = window
        history['offset'] = history.step - start
        history['included'] = (history.offset >= warmup) & history.benchmark_has_previous
        write_table(history, cache)
        tables.append(history)
        del state, result, solver
    timings = pd.concat(tables, ignore_index=True)
    summaries = []
    for window, group in timings[timings.included].groupby('window', sort=False):
        cold, warm = group.benchmark_cold_seconds, group.benchmark_warm_seconds
        base = group[['assignment_seconds', 'inner_seconds', 'outer_seconds', 'backward_seconds']].sum(axis=1)
        summaries.append(dict(window=window, pairs=len(group),
            cold_ms=1000 * cold.mean(), warm_ms=1000 * warm.mean(),
            cold_median_ms=1000 * cold.median(), warm_median_ms=1000 * warm.median(),
            implicit_speedup=cold.sum() / warm.sum(),
            cold_iterations=group.benchmark_cold_cg_iterations.mean(),
            warm_iterations=group.benchmark_warm_cg_iterations.mean(),
            cold_converged=bool(group.benchmark_cold_cg_converged.all()),
            warm_converged=bool(group.benchmark_warm_cg_converged.all()),
            cold_residual_max=group.benchmark_cold_cg_relative_residual.max(),
            warm_residual_max=group.benchmark_warm_cg_relative_residual.max(),
            solution_difference_max=group.benchmark_solution_relative_difference.max(),
            implicit_share_percent=100 * cold.sum() / (base + cold).sum(),
            estimated_step_speedup=(base + cold).sum() / (base + warm).sum()))
    summary = pd.DataFrame(summaries)
    write_table(timings, root / 'timings.csv')
    write_table(summary, root / 'summary.csv')
    return summary, timings, root
