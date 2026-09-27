import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from src.node_distances import array_digest
from src.ntk_transforms import fit_transform
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student


class AssignmentMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, material, chunk_size):
        ctx.save_for_backward(logits, material)
        ctx.chunk_size = chunk_size
        result = material.new_zeros(logits.shape[1], material.shape[1])
        for start in range(0, len(logits), chunk_size):
            end = start + chunk_size
            probability = logits[start:end].to(material.dtype).softmax(1)
            result.add_(probability.T @ material[start:end], alpha=1 / len(logits))
        return result

    @staticmethod
    def backward(ctx, gradient):
        logits, material = ctx.saved_tensors
        result = torch.empty_like(logits)
        for start in range(0, len(logits), ctx.chunk_size):
            end = start + ctx.chunk_size
            probability = logits[start:end].to(material.dtype).softmax(1)
            direction = material[start:end] @ gradient.T / len(logits)
            value = probability * (direction - (direction * probability).sum(1, keepdim=True))
            result[start:end] = value.to(logits.dtype)
        return result, None, None


def make_material(z, q):
    return torch.cat((z.new_ones(len(z), 1), z, q), dim=1)


def initial_logits(assignment, clusters, mixing=.05, dtype=torch.float32):
    if not 0 < mixing < 1:
        raise ValueError('mixing must lie strictly between zero and one')
    result = torch.full((len(assignment), clusters), np.log(mixing / clusters),
                         dtype=dtype, device=assignment.device)
    result.scatter_(1, assignment[:, None], float(np.log(1 - mixing + mixing / clusters)))
    return result


def decode_moments(moments, dimension):
    mass = moments[:, 0]
    centers = moments[:, 1:dimension + 1] / mass[:, None]
    labels = moments[:, dimension + 1:] / mass[:, None]
    return centers, labels, mass


def augmented(z):
    return torch.cat((z, z.new_ones(len(z), 1)), dim=1)


def ridge_head(centers, labels, mass, penalty):
    x = augmented(centers)
    gram = x.T @ (mass[:, None] * x)
    rhs = x.T @ (mass[:, None] * labels)
    system = gram + penalty * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(system, rhs)


def excess_objective(weight, reference, system):
    difference = weight - reference
    return (difference * (system @ difference)).sum() / 2


def teacher_ce(weight, z, q, temperature=1.):
    prediction = (z @ weight[:-1] + weight[-1]) / temperature
    return -(q * prediction.log_softmax(1)).sum(1).mean()


def moment_objective(moments, dimension, penalty, reference, system,
                     outer_loss='ridge', z=None, q=None, score_temperature=1.):
    centers, labels, mass = decode_moments(moments, dimension)
    weight = ridge_head(centers, labels, mass, penalty)
    if outer_loss == 'ridge':
        return excess_objective(weight, reference, system)
    if outer_loss == 'ce':
        return teacher_ce(weight, z, q, score_temperature)
    raise ValueError('outer_loss must be ridge or ce')


@torch.no_grad()
def ridge_metrics(z, q, y, ids, weight, penalty, test_ids=None, score_temperature=1.):
    prediction = augmented(z) @ weight
    mse = float((prediction - q).square().sum(1).mean())
    result = dict(val=100 * float((prediction[ids].argmax(1) == y[ids]).double().mean()),
                  val_mse=float((prediction[ids] - F.one_hot(y[ids], q.shape[1])).square().sum(1).mean()),
                  teacher_mse=mse, objective=mse / 2 + penalty * float(weight.square().sum()) / 2,
                  head_norm=float(weight.norm()), teacher_ce=float(teacher_ce(weight, z, q, score_temperature)),
                  val_ce=float(F.cross_entropy(prediction[ids] / score_temperature, y[ids])))
    if test_ids is not None:
        result['test'] = 100 * float((prediction[test_ids].argmax(1) == y[test_ids]).double().mean())
    return result


def optimize_assignment(z, q, assignment, penalty, reference, system, steps=1000, lr=.05,
                         mixing=.05, chunk_size=4096, log_every=10, save_assignment=True, folder=None,
                         outer_loss='ridge', score_temperature=1.):
    if steps < 0 or min(lr, chunk_size, log_every, penalty) <= 0:
        raise ValueError('Invalid optimizer settings')
    if outer_loss not in ('ridge', 'ce') or not np.isfinite(score_temperature) or score_temperature <= 0:
        raise ValueError('Invalid outer objective or score temperature')
    clusters = int(assignment.max()) + 1
    material = make_material(z, q)
    logits = initial_logits(assignment, clusters, mixing).requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=lr, eps=1e-12, foreach=False)
    history, best, best_step = [], float('inf'), 0
    best_logits = None
    started = time.perf_counter()
    for step in trange(steps + 1, desc=f'Soft assignment {outer_loss}'):
        optimizer.zero_grad(set_to_none=True)
        moments = AssignmentMoments.apply(logits, material, chunk_size)
        loss = moment_objective(moments, z.shape[1], penalty, reference, system,
                                outer_loss, z, q, score_temperature)
        value = float(loss.detach())
        if not np.isfinite(value) or bool((moments[:, 0] <= 0).any()):
            raise FloatingPointError('Nonfinite objective or numerically empty soft cell')
        if step == 0:
            initial_moments = moments.detach().clone()
            initial_value = value
            scale = max(value, 1e-12)
        if value < best:
            best, best_step = value, step
            best_moments = moments.detach().clone()
            if save_assignment:
                best_logits = logits.detach().clone()
        if step % log_every == 0 or step == steps:
            history.append(dict(step=step, J=value, best_J=best, relative_J=value / scale,
                                best_relative_J=best / scale, min_mass=float(moments[:, 0].min().detach()),
                                max_mass=float(moments[:, 0].max().detach()), seconds=time.perf_counter() - started))
            if folder is not None:
                pd.DataFrame(history).to_csv(Path(folder) / 'optimization.csv', index=False)
        if step == steps:
            break
        (loss / scale).backward()
        optimizer.step()
    result = dict(initial_moments=initial_moments.cpu(), best_moments=best_moments.cpu(),
                  final_moments=moments.detach().cpu(), best_step=best_step, history=history,
                  steps=steps, initial_J=initial_value, loss_scale=scale, best_J=best,
                  outer_loss=outer_loss, score_temperature=score_temperature)
    if save_assignment and folder is not None:
        torch.save(best_logits.cpu(), Path(folder) / 'best_assignment_logits.pt')
    return result


def run_soft_ridge(source, output_dir=None,
                   penalties=(1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
                   steps=1000, lr=.05, mixing=.05, chunk_size=4096, log_every=10,
                   final_seeds=tuple(range(100, 110)), save_assignment=True,
                   data_dir='/content/data/', device='cuda', evaluate_test=True,
                   outer_loss='ridge', score_temperatures=(1.,), comparison_source=None):
    source = Path(source)
    original = json.loads((source / 'config.json').read_text())
    if original.get('dataset') != 'arxiv' or not final_seeds or not penalties or any(not np.isfinite(p) or p <= 0 for p in penalties):
        raise ValueError('Require an Arxiv source and finite positive ridge penalties')
    if outer_loss not in ('ridge', 'ce') or not score_temperatures or any(not np.isfinite(t) or t <= 0 for t in score_temperatures):
        raise ValueError('Invalid outer objective or score temperatures')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adj = train['adj'].to_sparse_csr()
    digest = array_digest(train['x'].cpu().numpy(), train['y'].cpu().numpy(), adj.crow_indices().cpu().numpy(),
                          adj.col_indices().cpu().numpy(), adj.values().cpu().numpy(), mask.cpu().numpy(),
                          validation[1].cpu().numpy(), testing[1].cpu().numpy())
    if digest != original['data_digest']:
        raise ValueError('Source graph or splits differ')
    logits = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    params = original['params']
    q = (logits / params['T']).softmax(1).double()
    baseline = torch.load(source / 'baseline_partition.pt', map_location='cpu', weights_only=False)
    assignment = baseline['assignment'].to(device)
    clusters = len(baseline['counts'])
    if not torch.equal(torch.bincount(assignment).cpu(), baseline['counts'].long()):
        raise ValueError('Cached baseline assignments and counts differ')
    z, transform = fit_transform(h.double(), kind='rms')
    material = make_material(z, q)
    hard = material.new_zeros(clusters, material.shape[1]).index_add_(0, assignment, material) / len(z)
    if not torch.allclose(decode_moments(hard, z.shape[1])[1].float().cpu(), baseline['y'], atol=1e-5, rtol=1e-5):
        raise ValueError('Cached teacher labels differ from baseline cell means')
    config = dict(source=str(source), original=original, penalties=list(penalties), steps=steps, lr=lr,
                  mixing=mixing, chunk_size=chunk_size, log_every=log_every, final_seeds=list(final_seeds),
                  save_assignment=save_assignment, evaluate_test=evaluate_test,
                  outer_loss=outer_loss, score_temperatures=list(score_temperatures),
                  comparison_source=str(comparison_source) if comparison_source else None,
                  logits_digest=array_digest(logits.cpu().numpy()), assignment_digest=array_digest(assignment.cpu().numpy()),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    previous = None
    if comparison_source is not None:
        if outer_loss != 'ce':
            raise ValueError('Comparison reuse requires the CE outer objective')
        comparison_source = Path(comparison_source)
        previous_config = json.loads((comparison_source / 'config.json').read_text())
        for key in ('original', 'mixing', 'final_seeds', 'evaluate_test', 'logits_digest', 'assignment_digest'):
            if previous_config[key] != config[key]:
                raise ValueError(f'Comparison source differs in {key}')
        previous = torch.load(comparison_source / 'optimized.pt', map_location='cpu', weights_only=False)
        if previous.get('outer_loss', 'ridge') != 'ridge':
            raise ValueError('Comparison source must optimize the ridge objective')
        config['comparison_config'] = previous_config
        config['comparison_moments_digest'] = array_digest(previous['best_moments'].numpy())
    root = Path(output_dir or source / 'soft_ridge') / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    x = augmented(z)
    gram, rhs = x.T @ x / len(z), x.T @ q / len(z)
    val_ids = validation[1].nonzero().flatten()
    test_ids = testing[1].nonzero().flatten() if evaluate_test else None
    candidates, weights = [], []
    for penalty in penalties:
        system = gram + penalty * torch.eye(gram.shape[0], dtype=z.dtype, device=z.device)
        weight = torch.linalg.solve(system, rhs)
        weights.append(weight)
        candidates.append(dict(penalty=penalty, solve_residual=float((system @ weight - rhs).norm()),
                               **ridge_metrics(z, q, train['y'], val_ids, weight, penalty)))
    pd.DataFrame(candidates).to_csv(root / 'reference_grid.csv', index=False)
    choice = min(range(len(candidates)), key=lambda i: (-candidates[i]['val'], candidates[i]['val_mse']))
    penalty, reference = penalties[choice], weights[choice]
    if previous is not None:
        previous_penalty = float(pd.read_csv(comparison_source / 'reference.csv').penalty.iloc[0])
        if penalty != previous_penalty:
            raise ValueError('Inner ridge penalty differs from the comparison source')
    with torch.no_grad():
        scores = augmented(z[val_ids]) @ reference
        temperatures = [dict(score_temperature=t, val_ce=float(F.cross_entropy(scores / t, train['y'][val_ids])))
                        for t in score_temperatures]
    pd.DataFrame(temperatures).to_csv(root / 'temperature_grid.csv', index=False)
    temperature = min(temperatures, key=lambda row: row['val_ce'])['score_temperature']
    system = gram + penalty * torch.eye(gram.shape[0], dtype=z.dtype, device=z.device)
    reference_values = ridge_metrics(z, q, train['y'], val_ids, reference, penalty, test_ids, temperature)
    pd.DataFrame([dict(penalty=penalty, at_lower_boundary=penalty == min(penalties),
                       solve_residual=candidates[choice]['solve_residual'], score_temperature=temperature,
                       **reference_values)]).to_csv(root / 'reference.csv', index=False)
    torch.save(dict(weight=reference.cpu(), transform=transform.state_dict(), penalty=penalty,
                    score_temperature=temperature), root / 'reference.pt')
    optimized_path = root / 'optimized.pt'
    if optimized_path.exists():
        optimized = torch.load(optimized_path, map_location='cpu', weights_only=False)
    else:
        optimized = optimize_assignment(z, q, assignment, penalty, reference, system, steps, lr, mixing,
                                         chunk_size, log_every, save_assignment, root, outer_loss, temperature)
        torch.save(optimized, optimized_path)
    pd.DataFrame(optimized['history']).to_csv(root / 'optimization.csv', index=False)
    summaries, student_tables = [], []
    settings = {k: original[k] for k in ('epochs', 'eval_every', 'hidden')}
    conditions = [('hard_baseline', hard), ('soft_initial', optimized['initial_moments'].to(device))]
    if previous is not None:
        if not torch.allclose(optimized['initial_moments'], previous['initial_moments'], atol=1e-10, rtol=1e-8):
            raise ValueError('Soft initialization differs from the comparison source')
        conditions.append(('ridge_optimized', previous['best_moments'].to(device)))
    conditions.append(('ce_optimized' if outer_loss == 'ce' else 'soft_optimized', optimized['best_moments'].to(device)))
    for method, moments in conditions:
        centers, labels, mass = decode_moments(moments, z.shape[1])
        weight = ridge_head(centers, labels, mass, penalty)
        values = ridge_metrics(z, q, train['y'], val_ids, weight, penalty, test_ids, temperature)
        excess = float(excess_objective(weight, reference, system))
        cx = (centers * transform.scale + transform.output_center + transform.center).float()
        torch.save(dict(x=cx.cpu(), y=labels.float().cpu(), mass=mass.cpu(), weight=weight.cpu()), root / f'{method}.pt')
        final_path = root / f'{method}_students.csv'
        if previous is not None and method != 'ce_optimized' and not final_path.exists():
            old_method = 'soft_optimized' if method == 'ridge_optimized' else method
            pd.read_csv(comparison_source / f'{old_method}_students.csv').to_csv(final_path, index=False)
        if final_path.exists():
            final = pd.read_csv(final_path)
            if sorted(final.seed.tolist()) != sorted(final_seeds):
                raise ValueError('Cached student seeds differ from requested seeds')
        else:
            records = []
            for seed in final_seeds:
                val, test, epoch = _train_student(cx, labels.float(), validation, params, seed,
                                                  {**settings, 'loss_weighting': 'mass'},
                                                  testing=testing if evaluate_test else None, counts=mass)
                records.append(dict(seed=seed, val=100 * val, test=100 * test if test is not None else np.nan, epoch=epoch))
                pd.DataFrame(records).to_csv(root / f'{method}_students.partial.csv', index=False)
            final = pd.DataFrame(records)
            final.to_csv(final_path, index=False)
        student_tables.append(final.assign(method=method))
        summaries.append(dict(method=method, nodes=clusters, penalty=penalty, excess_objective=excess,
                              score_temperature=temperature, teacher_ce=values['teacher_ce'], ridge_val_ce=values['val_ce'],
                              ridge_val=values['val'], ridge_test=values.get('test', np.nan),
                              teacher_mse=values['teacher_mse'], head_distance=float((weight - reference).norm()),
                              val=final.val.mean(), val_std=final.val.std(ddof=0), test=final.test.mean(),
                              test_std=final.test.std(ddof=0), min_mass=float(mass.min()), max_mass=float(mass.max()),
                              effective_cells=float(1 / mass.square().sum()),
                              best_step=(previous['best_step'] if method == 'ridge_optimized' else
                                         optimized['best_step'] if method in ('soft_optimized', 'ce_optimized') else 0),
                              output_dir=str(root)))
        pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
        pd.concat(student_tables).to_csv(root / 'students.csv', index=False)
    return pd.DataFrame(summaries)


def run_soft_ridge_ce(previous_run, output_dir=None, score_temperatures=(.01, .02, .05, .1, .2, .5, 1., 2.),
                      steps=None, lr=None, chunk_size=None, save_assignment=True,
                      data_dir='/content/data/', device='cuda'):
    previous_run = Path(previous_run)
    config = json.loads((previous_run / 'config.json').read_text())
    penalty = float(pd.read_csv(previous_run / 'reference.csv').penalty.iloc[0])
    return run_soft_ridge(config['source'], output_dir=output_dir, penalties=[penalty],
                           steps=config['steps'] if steps is None else steps,
                           lr=config['lr'] if lr is None else lr, mixing=config['mixing'],
                           chunk_size=config['chunk_size'] if chunk_size is None else chunk_size,
                           log_every=config['log_every'], final_seeds=config['final_seeds'],
                           save_assignment=save_assignment, data_dir=data_dir, device=device,
                           evaluate_test=config['evaluate_test'], outer_loss='ce',
                           score_temperatures=score_temperatures, comparison_source=previous_run)
