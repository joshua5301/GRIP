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


def compare_newton_performance(previous_run, output_dir, steps=1000,
                               checkpoint_steps=(0, 100, 300, 500, 750, 1000),
                               search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                               data_dir='/content/data/', device='cuda'):
    from src.assignment_sweep import representative
    from src.condensation_diagnostics import fit_gcn_diagnostic
    from src.node_distances import array_digest
    from src.ntk_transforms import FeatureTransform
    from src.risk_experiment import _prepare_dataset

    checkpoints = sorted({0, steps, *checkpoint_steps})
    if steps < 1 or any(s < 0 or s > steps for s in checkpoints) or not search_seeds or not final_seeds:
        raise ValueError('Invalid comparison budget or evaluation seeds')
    source = Path(previous_run)
    prior = json.loads((source / 'config.json').read_text())
    choice = json.loads((source / 'selected_low_rank.json').read_text())
    candidate = source / f'candidate_{int(choice["candidate"]):04d}'
    params = json.loads((candidate / 'params.json').read_text())
    options = dict(prior['solver'], solver_mode='exact', implicit_warm_start=True)
    options.pop('inner_method', None)
    config = dict(source=str(source), prior=prior, params=params, steps=steps,
                  checkpoints=checkpoints, search_seeds=list(search_seeds),
                  final_seeds=list(final_seeds), solver=options, version=1)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / 'config.json')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train_mask, validation, testing, unused = _prepare_dataset(prior['dataset'], data_dir, device)
    del unused
    adj = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adj.crow_indices(), adj.col_indices(), adj.values(),
        train_mask, validation[1], testing[1])))
    if digest != prior['data_digest']:
        raise ValueError('Dataset differs from the source experiment')
    saved = torch.load(source / 'inputs.pt', map_location=device, weights_only=False)
    z, assignment = saved['z'], saved['assignment']
    transform = FeatureTransform(**saved['transform'])
    del saved
    teacher = torch.load(source / 'teacher.pt', map_location=device, weights_only=False)
    q = (teacher['logits'] / params['T']).softmax(1).double()
    del teacher
    masks = dict(train=train_mask, val=validation[1], test=testing[1])
    settings = dict(epochs=prior['epochs'], eval_every=prior['eval_every'], hidden=prior['hidden'],
                    dropout=prior['dropout'], lr=prior['student_lr'], weight_decay=prior['weight_decay'])
    summaries, curves, students, histories = [], [], [], []
    initial = None
    for method in ('lbfgs', 'newton_first'):
        folder = root / method
        folder.mkdir(exist_ok=True)
        complete = folder / 'complete.json'
        if not complete.exists():
            resume = folder / 'resume.pt'
            state = torch.load(resume, map_location='cpu', weights_only=False) if resume.exists() else None
            result = optimize_ce_assignment(z, q, assignment, penalty=params['penalty'],
                lr=params['assignment_lr'], steps=steps, mixing=prior['mixing'], factor_seed=prior['seed'],
                folder=folder, checkpoint_steps=checkpoints, resume_state=state, save_resume=True,
                save_assignment=False, inner_method=method, **family_options(params), **options)
            save_json(dict(steps=steps), complete)
            del result, state
        history = pd.read_csv(folder / 'optimization.csv')
        histories.append(history.assign(method=method))
        moments0 = torch.load(folder / 'checkpoints/step_000000.pt', map_location='cpu',
                              weights_only=False)['moments']
        if initial is not None and not torch.equal(initial, moments0):
            raise ValueError('Initial moments differ between solvers')
        initial = moments0

        def evaluate(step, seeds, final=False):
            snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt',
                                  map_location='cpu', weights_only=False)
            cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
            cache = folder / ('final' if final else 'search') / f'step_{step:06d}'
            scores = pd.DataFrame([fit_gcn_diagnostic(cx, cy, mass, graph, q,
                masks if final else {k: masks[k] for k in ('train', 'val')}, seed,
                folder=cache, **settings) for seed in seeds])
            return scores

        method_curve = []
        for step in checkpoints:
            scores = evaluate(step, search_seeds)
            method_curve.append(dict(method=method, step=step, val=scores.val_acc.mean(),
                                     val_std=scores.val_acc.std(ddof=0)))
        curve = pd.DataFrame(method_curve)
        curves.extend(method_curve)
        selected = int(curve.sort_values(['val', 'step'], ascending=[False, True]).iloc[0].step)
        update_time = history.loc[history.step < steps,
            ['assignment_seconds', 'inner_seconds', 'outer_seconds', 'implicit_seconds', 'backward_seconds']].sum().sum()
        for phase, step in [('initial', 0), ('fixed_budget', steps), ('selected', selected)]:
            scores = evaluate(step, final_seeds, final=True)
            students.extend(scores.assign(method=method, phase=phase, step=step).to_dict('records'))
            summaries.append(dict(method=method, phase=phase, step=step,
                search_val=float(curve.loc[curve.step == step, 'val'].iloc[0]),
                final_val=scores.val_acc.mean(), final_val_std=scores.val_acc.std(ddof=0),
                test_mean=scores.test_acc.mean(), test_std=scores.test_acc.std(ddof=0),
                update_seconds=update_time, condensation_seconds=float(history.seconds.iloc[-1])))
        write_table(pd.DataFrame(summaries), root / 'summary.csv')
        write_table(pd.DataFrame(curves), root / 'search.csv')
        write_table(pd.DataFrame(students), root / 'final_students.csv')
    paired = pd.DataFrame(students).pivot(index=['phase', 'seed'], columns='method',
                                        values=['val_acc', 'test_acc'])
    differences = pd.DataFrame({key + '_delta_pp': paired[key]['newton_first'] - paired[key]['lbfgs']
                               for key in ('val_acc', 'test_acc')}).reset_index()
    write_table(differences, root / 'paired_differences.csv')
    write_table(pd.concat(histories, ignore_index=True), root / 'optimization.csv')
    return pd.DataFrame(summaries), pd.DataFrame(curves), differences, root
