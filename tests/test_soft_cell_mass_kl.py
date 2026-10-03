"""New activated-only checkpoint lineage regression; no heads or P updates."""
import copy

import pytest
import torch

from src.soft_cell_mass_kl import (
    MODE, OBJECTIVE, POLICY, _factor_digests, attach_resume, attach_snapshot, validate_resume,
)


def test_resumed_soft_mass_kl_rejects_scale_history_and_terminal_factor_corruption():
    nodes, cells, rank, dimension, classes = 5, 3, 2, 2, 2
    native = [torch.zeros(nodes, rank, dtype=torch.float32), torch.arange(cells * rank).reshape(cells, rank).float()]
    context = dict(schema=1, policy=POLICY, source_refs={},
                   native_parameter_digests=_factor_digests(native, nodes, cells, rank))
    config = dict(soft_cell_mass_KL_mode=MODE, soft_cell_mass_KL_context=context)
    mass = torch.tensor([.2, .3, .5], dtype=torch.float64)
    features = torch.tensor([[1., 2.], [2., 1.], [3., 4.]], dtype=torch.float64)
    labels = torch.tensor([[.3, .7], [.5, .5], [.8, .2]], dtype=torch.float64)
    moments = torch.cat((mass[:, None], mass[:, None]*features, mass[:, None]*labels), 1)
    theta = torch.zeros(classes, dimension + 1, dtype=torch.float64)
    terminal = [native[0] + .01, native[1] + .02]
    # Deliberately synthetic scalar ledger fixture: lineage only, not a measured
    # Omega(M) value. The separately frozen DR oracle qualifies numerical math.
    CE0, Omega0 = 1.2, .7
    snapshots, history = {}, []
    for step, parameters, CE, Omega in ((0, native, CE0, Omega0), (1, terminal, 1.1, .6)):
        objective = CE/CE0 + Omega/Omega0
        snap = dict(step=step, moments=moments.clone(), theta=theta.clone(), teacher_ce=CE, J_exact=True)
        attach_snapshot(snap, context, config, parameters, CE0, Omega0, Omega, objective)
        snapshots[step] = snap
        history.append(dict(step=step, J=CE, teacher_ce=CE, J_exact=True, inner_converged=True,
            soft_cell_mass_KL_CE0=CE0, soft_cell_mass_KL_Omega0=Omega0,
            soft_cell_mass_KL_Omega=Omega, objective=objective,
            normalized_objective=objective, objective_name=OBJECTIVE))
    state = dict(config=config, step=1, snapshots=snapshots, history=history,
                 initial_moments=moments.clone(), theta=theta.clone(), parameters=terminal,
                 scale=CE0, best_step=1, best=history[1]["objective"],
                 best_moments=moments.clone(), best_theta=theta.clone())
    attach_resume(state, context, native, CE0, Omega0)
    check = lambda value: validate_resume(value, context, config, 25, nodes, cells, rank, dimension, classes)
    check(state)
    altered = copy.deepcopy(state)
    altered["soft_cell_mass_KL_Omega0"] = .8
    with pytest.raises(ValueError):
        check(altered)
    altered = copy.deepcopy(state)
    altered["history"][1]["normalized_objective"] += .01
    with pytest.raises(ValueError):
        check(altered)
    altered = copy.deepcopy(state)
    altered["parameters"][0][0, 0] += .01
    with pytest.raises(ValueError):
        check(altered)

    altered = copy.deepcopy(state)
    altered["soft_cell_mass_KL_context"]["policy"] = dict(POLICY, coefficient=2)
    with pytest.raises(ValueError):
        check(altered)
    altered = copy.deepcopy(state)
    altered["snapshots"][0]["soft_cell_mass_KL_parameters"][1][0, 0] += .01
    with pytest.raises(ValueError):
        check(altered)
