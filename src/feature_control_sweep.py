import json
from pathlib import Path

import pandas as pd
import torch

from src.assignment_sweep import run_assignment_sweep
from src.risk_experiment import _fingerprint


def paired_grid(assignment, direct):
    keys = ['candidate', 'step', 'dropout', 'gamma', 'T', 'penalty', 'assignment_lr']
    columns = keys + ['val', 'val_std', 'teacher_ce']
    paired = assignment[columns].merge(direct[columns], on=keys, suffixes=('_B', '_C'),
                                       how='outer', validate='one_to_one', indicator=True)
    if not (paired['_merge'] == 'both').all():
        raise ValueError('B and C must have the same evaluated configurations')
    paired = paired.drop(columns='_merge')
    paired['val_delta_C_minus_B'] = paired.val_C - paired.val_B
    paired['ce_delta_C_minus_B'] = paired.teacher_ce_C - paired.teacher_ce_B
    return paired


def run_feature_control_sweep(dataset, ratios, output_dir, space, gammas, **options):
    if {'feature_control', 'full_grid', 'initialization_source'} & options.keys():
        raise ValueError('This comparison always runs both fixed-target controls on the full grid')
    tables, source = [], None
    for control, method in [('assignment', 'B'), ('direct', 'C')]:
        table = run_assignment_sweep(
            dataset=dataset, ratios=ratios, output_dir=str(Path(output_dir) / method),
            space=space, gammas=gammas, full_grid=True, feature_control=control,
            initialization_source=source, **options)
        tables.append(table.assign(method=method))
        source = table.output_dir.iloc[0]
    results = pd.concat(tables, ignore_index=True)
    root = Path(output_dir) / ('comparison_' + _fingerprint(results.output_dir.tolist()))
    root.mkdir(parents=True, exist_ok=True)
    roots = {method: Path(group.output_dir.iloc[0]) for method, group in results.groupby('method')}
    paired, differences, checks = [], [], []
    for ratio in ratios:
        folders = {method: path / f'ratio_{ratio:g}' for method, path in roots.items()}
        grids = {method: pd.read_csv(path / 'full_grid.csv') for method, path in folders.items()}
        paired.append(paired_grid(grids['B'], grids['C']).assign(dataset=dataset, ratio=ratio))
        for candidate in grids['B'].drop_duplicates(['gamma', 'T']).candidate:
            initials = [torch.load(path / f'candidate_{candidate:03d}' / 'checkpoints' / 'step_000000.pt',
                                   map_location='cpu', weights_only=False)['moments']
                        for path in folders.values()]
            error = float((initials[0] - initials[1]).abs().max())
            if not torch.allclose(initials[0], initials[1], rtol=1e-10, atol=1e-12):
                raise ValueError('B and C did not start from the same representatives')
            checks.append(dict(ratio=ratio, candidate=candidate, initial_max_difference=error))
        final = {method: pd.read_csv(path / 'final_students.csv') for method, path in folders.items()}
        joined = final['B'].merge(final['C'], on='seed', suffixes=('_B', '_C'), validate='one_to_one')
        for metric in ('val', 'test'):
            delta = joined[f'{metric}_C'] - joined[f'{metric}_B']
            differences.append(dict(dataset=dataset, ratio=ratio, metric=metric,
                                    delta_C_minus_B=delta.mean(), paired_std=delta.std(ddof=0),
                                    n_seeds=len(delta)))
    results['comparison_dir'] = str(root)
    results.to_csv(root / 'summary.csv', index=False)
    pd.concat(paired, ignore_index=True).to_csv(root / 'matched_grid.csv', index=False)
    pd.DataFrame(differences).to_csv(root / 'final_deltas.csv', index=False)
    pd.DataFrame(checks).to_csv(root / 'initialization_checks.csv', index=False)
    (Path(output_dir) / 'latest_comparison.json').write_text(json.dumps(dict(path=str(root))), encoding='utf-8')
    return results
