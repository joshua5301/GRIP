"""Synthetic-only citation GCN context/cache tests; no dataset or GPU fitting."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import src.citation_gcn_teacher as teacher
import src.citation_search as search
from src.data import BUDGET
from src.gcn_teacher import _raw_forward
from src.io import _fingerprint, cpu_state, save_json, save_state
from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.models import GCN
from src.moments import decode_moments, make_material
from src.soft_ce_partition import solve_inner_newton_first
from src.transforms import fit_transform


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setitem(BUDGET, ('cora', .013), 3)
    g = torch.Generator().manual_seed(814)
    graph = dict(x=torch.randn(12, 4, generator=g), y=torch.tensor([0, 1] * 6),
                 adj=(torch.eye(12) * .7 + torch.ones(12, 12) * .025).to_sparse_csr())
    train, val, test = torch.arange(12) < 4, (torch.arange(12) >= 4) & (torch.arange(12) < 8), torch.arange(12) >= 8
    h = graph['adj'] @ (graph['adj'] @ graph['x'])
    calls = dict(data=0, teacher=0, optimizer=0, student=0, toy_heads=0)
    def data(*args, **kwargs):
        calls['data'] += 1
        return graph, train, (graph, val), (graph, test), h
    monkeypatch.setattr(search, '_prepare_dataset', data)
    import src.data as module
    monkeypatch.setattr(module, '_prepare_dataset', data)
    config = search._legacy_teacher_config('cora', .013, graph, train, (graph, val), (graph, test), 'default')
    root = tmp_path / 'cora' / 'ratio_0.013' / _fingerprint(config)
    root.mkdir(parents=True)
    save_json(config, root / 'config.json')
    h = search.fixed_propagated_features(h, config, root)
    z, transform = fit_transform(h.double())
    assignment = torch.arange(12) % 3
    save_state(dict(z=z, assignment=assignment, transform=vars(transform)), root / 'inputs_0.pt')
    save_state(dict(logits=torch.zeros(12, 2, dtype=torch.double), gamma=.01), root / 'teacher.pt')
    candidate = dict(method='low_rank', width=0, lr=.01, T=.3, rank=2, penalty=.1,
                     initialization='teacher_balanced', alpha=.3, inner_loss_weighting='uniform')
    key = _fingerprint(candidate); (root / key).mkdir()
    save_json(candidate, root / key / 'candidate.json')
    init = dict(mode='teacher_balanced', alpha=.3, T=.3, seed=0)
    assignment_path = root / f'assignment_{_fingerprint(init)}.pt'
    save_state(assignment, assignment_path)
    pin = dict(policy='frozen_relu', source_root=str(root.resolve()),
               source_config_sha256=teacher.file_digest(root / 'config.json'),
               source_H_sha256=teacher.file_digest(root / 'propagated_H.pt'),
               source_inputs_path=str((root / 'inputs_0.pt').resolve()),
               source_inputs_sha256=teacher.file_digest(root / 'inputs_0.pt'),
               source_ReLU_teacher_sha256=teacher.file_digest(root / 'teacher.pt'),
               source_hard_assignment_path=str(assignment_path.resolve()),
               source_hard_assignment_sha256=teacher.file_digest(assignment_path),
               origin_initialization='teacher_balanced', origin_alpha=.3, origin_T=.3,
               origin_condensation_seed=0, origin_candidate_id=key,
               origin_candidate_sha256=teacher.file_digest(root / key / 'candidate.json'))
    def fitted(graph, train, validation, folder, **kwargs):
        calls['teacher'] += 1
        path = Path(folder) / 'teacher.pt'
        if path.exists():
            return load(path)
        assert kwargs['epochs'] == 200 and kwargs['seed'] == 0
        kwargs['guard']()
        torch.manual_seed(5)
        model = GCN(4, 256, 2, 2, .5).eval()
        with torch.no_grad():
            logits = _raw_forward(model, graph['x'], graph['adj'])
        scores, labels = logits[validation[1]], graph['y'][validation[1]]
        metric = dict(epoch=1, val_acc=100 * float((scores.argmax(1) == labels).double().mean()),
                      val_ce=float(F.cross_entropy(scores, labels)), val_nodes=len(labels))
        recipe = teacher._teacher_recipe(graph, train, validation)
        result = dict(training_complete=True, fingerprint=_fingerprint(recipe), recipe=recipe, logits=logits,
                      selected_state=cpu_state(model.state_dict()), selected_validation=metric,
                      history=[dict(metric, epoch=e) for e in range(1, 201)],
                      timings=dict(training_seconds=0., validation_seconds=0., source_logits_seconds=0.))
        path.parent.mkdir(parents=True, exist_ok=True); save_state(result, path)
        return result
    monkeypatch.setattr(teacher, 'fit_gcn_teacher', fitted)
    def optimize(z, q, assignment, **kwargs):
        calls['optimizer'] += 1
        folder = Path(kwargs['folder']); folder.mkdir(parents=True, exist_ok=True)
        u, v = initialize_factors(assignment, 3, 2, 0)
        moments = LowRankMoments.apply(u, v, assignment, make_material(z, q), .05, 4096).detach()
        x, y, mass = decode_moments(moments, 4)
        fitted = solve_inner_newton_first(x, y, torch.full_like(mass, 1 / len(mass)), .1, grad_tol=1e-9)
        calls['toy_heads'] += 1
        assert fitted['inner_converged']
        snapshots = {step: dict(step=step, moments=moments, theta=fitted['theta'], teacher_ce=1.,
                                inner_grad_max=fitted['inner_grad_max'], J_exact=True)
                     for step in (0, kwargs['steps'])}
        (folder / 'checkpoints').mkdir(exist_ok=True)
        for step, snapshot in snapshots.items():
            save_state(snapshot, folder / 'checkpoints' / f'step_{step:06d}.pt')
        save_state(dict(config=teacher.expected_resume_config(candidate, z, q, assignment, 0),
                        step=kwargs['steps'], parameters=[u, v], initial_moments=moments, snapshots=snapshots), folder / 'resume.pt')
    monkeypatch.setattr(search, 'optimize_ce_assignment', optimize)
    def student(*args, **kwargs):
        calls['student'] += 1
        return dict(epoch=1, val_acc=50., val_ce=1.)
    monkeypatch.setattr(search, 'fit_gcn_diagnostic', student)
    return dict(graph=graph, train=train, validation=(graph, val), h=h, config=config, source=root,
                pin=pin, candidate=candidate, output=tmp_path, calls=calls)


def prepare(toy, **extra):
    return teacher.prepare_job('cora', .013, toy['output'], toy['pin'], device='cpu', **extra)


def screen(toy, **extra):
    return search.run_screen('cora', .013, toy['output'], [toy['candidate']], steps=25, checkpoints=[0],
                            student_seeds=(2600,), dropout=0, epochs=1, device='cpu',
                            teacher_backend=teacher.BACKEND, initialization_source=toy['pin'], **extra)


@pytest.mark.parametrize('change', ['backend', 'source_missing', 'pin_keys', 'seed', 'NTK', 'mass', 'target_mix',
                                  'method', 'inner', 'mix', 'T', 'alpha', 'embedded'])
def test_unsupported_rejects_before_dataset_and_writes(toy, change):
    candidate = dict(toy['candidate']); pin = dict(toy['pin']); backend = teacher.BACKEND; kernel = 'relu'; seed = 0
    if change == 'backend': backend = 'gcn'
    elif change == 'source_missing': pin = None
    elif change == 'pin_keys': pin['free_assignment'] = 'bad'
    elif change == 'seed': seed = 1
    elif change == 'NTK': kernel = 'relu_ntk1_diagmatch_v1'
    elif change == 'mass': candidate['mass_mode'] = 'uniform'
    elif change == 'target_mix': candidate['train_target_mix'] = .1
    elif change == 'method': candidate['method'] = 'nystrom'
    elif change == 'inner': candidate['inner_loss_weighting'] = 'mass'
    elif change == 'mix': candidate['mixing'] = .1
    elif change == 'T': candidate['T'] = 1.
    elif change == 'alpha': candidate['alpha'] = 1.
    else: candidate['teacher_backend'] = backend
    before = files(toy['output'])
    with pytest.raises(ValueError):
        search.run_screen('cora', .013, toy['output'], [candidate], teacher_backend=backend,
                          initialization_source=pin, teacher_kernel=kernel, condensation_seed=seed, device='cpu')
    assert toy['calls']['data'] == 0 and files(toy['output']) == before


def test_frozen_recipe_guard_preload(toy):
    wrong = teacher.recipe(); wrong['epochs'] = 199
    before = files(toy['output'])
    with pytest.raises(ValueError): prepare(toy, teacher_recipe=wrong)
    assert toy['calls']['data'] == 0 and files(toy['output']) == before


def test_legacy_default_and_explicit_none_same_identity(toy, monkeypatch):
    monkeypatch.setattr(search, 'teacher_logits', lambda *a, **k: (torch.zeros(12, 2, dtype=torch.double), .01))
    _, first = search.run_screen('cora', .013, toy['output'], [toy['candidate']], steps=0, student_seeds=(1,), device='cpu', epochs=1)
    _, second = search.run_screen('cora', .013, toy['output'], [toy['candidate']], steps=0, student_seeds=(1,), device='cpu', epochs=1,
                                  teacher_backend=None, initialization_source=None, teacher_kernel='relu')
    assert first == second == toy['source'] and 'teacher_backend' not in json.loads((first / 'config.json').read_text())


def test_complete_prepare_replay_and_source_only_copy(toy):
    old = files(toy['source']); result = prepare(toy); root = Path(result['root'])
    selected, config, (h, inputs, assignment), proof = teacher.validate_root(root, toy['graph'], toy['train'], toy['validation'], toy['h'])
    assert files(toy['source']) == old and root != toy['source']
    assert selected['logits'].dtype == torch.float32 and config['gcn_context']['q_formation'] == teacher.Q_FORMATION
    assert torch.equal(h, toy['h']) and torch.equal(inputs['z'], load(toy['source'] / 'inputs_0.pt')['z'])
    assert torch.equal(assignment, load(toy['pin']['source_hard_assignment_path']))
    assert proof['q_digest'] == teacher._digest((selected['logits'].double() / .3).softmax(1))
    assert not list(root.glob('assignment_*.pt')) and not list(root.glob('*map*')) and not list(root.glob('*phi*'))


def test_native_logits_double_before_q_and_cached_bypass(toy):
    root = Path(prepare(toy)['root']); selected = load(root / 'teacher.pt'); _, generated = screen(toy)
    assert generated == root and toy['calls']['optimizer'] == 1
    state = load(root / _fingerprint(toy['candidate']) / 'condensation_0' / 'resume.pt')
    z = load(toy['source'] / 'inputs_0.pt')['z']; assignment = load(toy['pin']['source_hard_assignment_path'])
    q = (selected['logits'].double() / .3).softmax(1)
    assert state['config'] == teacher.expected_resume_config(toy['candidate'], z, q, assignment, 0)
    assert float((q.sum(1) - 1).abs().max()) < 1e-14
    assert not torch.equal(q, (selected['logits'] / .3).softmax(1).double())
    old = files(root); screen(toy)
    assert toy['calls']['optimizer'] == 1 and files(root) == old


@pytest.mark.parametrize('change', ['logits', 'weights', 'history_short', 'history_nan', 'history_bool', 'history_tie',
                                  'selected_epoch', 'recipe', 'complete', 'native_dtype', 'state_dtype', 'certificate', 'q_digest'])
def test_hot_teacher_corruption_rejects_before_fit_optimizer_student_and_mutation(toy, change):
    root = Path(prepare(toy)['root']); path = root / 'gcn_teacher' / 'teacher.pt'; state = load(path)
    if change == 'logits': state['logits'][0, 0] += 1
    elif change == 'weights': next(iter(state['selected_state'].values())).view(-1)[0] += 1
    elif change == 'history_short': state['history'].pop()
    elif change == 'history_nan': state['history'][2]['val_ce'] = float('nan')
    elif change == 'history_bool': state['history'][2]['val_nodes'] = True
    elif change == 'history_tie': state['selected_validation'] = state['history'][1]
    elif change == 'selected_epoch': state['selected_validation']['epoch'] = 2
    elif change == 'recipe': state['recipe']['settings']['epochs'] = 199
    elif change == 'complete': state['training_complete'] = False
    elif change == 'native_dtype': state['logits'] = state['logits'].double()
    elif change == 'state_dtype': state['selected_state'] = {k: v.double() for k, v in state['selected_state'].items()}
    else:
        path = root / 'teacher.pt'; state = load(path)
        if change == 'certificate': state['schema'] = True
        else: state['q_digest'] = 'bad'
    save_state(state, path); before = files(root); counts = dict(toy['calls'])
    with pytest.raises(ValueError): prepare(toy)
    assert files(root) == before and toy['calls']['teacher'] == counts['teacher']
    with pytest.raises(ValueError): screen(toy)
    assert files(root) == before and toy['calls']['optimizer'] == counts['optimizer'] and toy['calls']['student'] == counts['student']


def test_native_complete_interruption_before_certificate_recovers_cached_no_refit(toy, monkeypatch):
    original = teacher.save_state
    def stopped(state, path):
        if Path(path).parent.name != 'gcn_teacher' and Path(path).name == 'teacher.pt':
            raise InterruptedError('after native completion')
        return original(state, path)
    monkeypatch.setattr(teacher, 'save_state', stopped)
    with pytest.raises(InterruptedError): prepare(toy)
    root = next(p.parent for p in toy['output'].rglob('source_binding.json'))
    assert (root / 'gcn_teacher' / 'teacher.pt').exists() and not (root / 'teacher.pt').exists()
    with pytest.raises(ValueError, match='certificate'): screen(toy)
    monkeypatch.setattr(teacher, 'save_state', original)
    result = prepare(toy); assert result['cached'] and (root / 'teacher.pt').exists()


@pytest.mark.parametrize('change', ['H', 'inputs', 'assignment', 'config', 'candidate', 'teacher', 'external_path', 'source_sha'])
def test_frozen_source_tampering_rejected_before_new_root_or_fit(toy, change):
    pin = toy['pin']
    if change == 'external_path': pin['source_hard_assignment_path'] = str(toy['source'] / 'inputs_0.pt')
    elif change == 'source_sha': pin['source_H_sha256'] = 'bad'
    else:
        key = dict(H='propagated_H.pt', inputs='inputs_0.pt', assignment=Path(pin['source_hard_assignment_path']).name,
                   config='config.json', candidate=f"{pin['origin_candidate_id']}/candidate.json", teacher='teacher.pt')[change]
        path = toy['source'] / key; path.write_bytes(path.read_bytes() + b'corrupt')
    before = files(toy['output'])
    with pytest.raises((ValueError, json.JSONDecodeError)): prepare(toy)
    assert files(toy['output']) == before and toy['calls']['teacher'] == 0


@pytest.mark.parametrize('change', ['node_order', 'train_mask', 'val_mask', 'train_labels', 'val_labels', 'graph', 'dtype'])
def test_current_graph_input_changes_reject_hot_cache(toy, change):
    root = Path(prepare(toy)['root']); before = files(root)
    graph = {k: v.clone() for k, v in toy['graph'].items()}; train = toy['train'].clone(); val = toy['validation'][1].clone()
    if change == 'node_order': graph['x'] = graph['x'].flip(0)
    elif change == 'train_mask': train[0] = False
    elif change == 'val_mask': val[4] = False
    elif change == 'train_labels': graph['y'][0] = 1
    elif change == 'val_labels': graph['y'][4] = 1
    elif change == 'graph': graph['adj'].values()[0] += .01
    else: graph['x'] = graph['x'].double()
    with pytest.raises(ValueError): teacher.validate_root(root, graph, train, (graph, val), toy['h'])
    assert files(root) == before


@pytest.mark.parametrize('change', ['config', 'parameters', 'snapshot', 'theta', 'initial', 'binding', 'orphan', 'candidate'])
def test_completed_condensation_corruption_rejected_before_scientific_bypass(toy, change):
    root = Path(prepare(toy)['root']); screen(toy)
    folder = root / _fingerprint(toy['candidate']) / 'condensation_0'; path = folder / 'resume.pt'; state = load(path)
    if change == 'config': state['config']['inner_loss_weighting'] = 'mass'
    elif change == 'parameters': state['parameters'][0][0, 0] += .1
    elif change == 'initial': state['initial_moments'][0, 0] += .1
    elif change in ('snapshot', 'theta'):
        path = folder / 'checkpoints' / 'step_000025.pt'; state = load(path)
        state['moments' if change == 'snapshot' else 'theta'][0, 0] += .1
    elif change == 'orphan': path.unlink()
    elif change == 'binding':
        path = folder / 'gcn_binding.json'; state = json.loads(path.read_text()); state['q_digest'] = 'bad'; save_json(state, path)
    elif change == 'candidate':
        path = folder.parent / 'candidate.json'; state = json.loads(path.read_text()); state['T'] = 1.; save_json(state, path)
    if change not in ('orphan', 'binding', 'candidate'): save_state(state, path)
    before = files(root); counts = dict(toy['calls'])
    with pytest.raises(ValueError): screen(toy)
    assert files(root) == before and toy['calls']['optimizer'] == counts['optimizer'] and toy['calls']['student'] == counts['student']


def test_selected_test_new_backend_guard_before_selection_writes(toy):
    root = Path(prepare(toy)['root']); screen(toy)
    raw = load(root / 'teacher.pt'); raw['q_digest'] = 'wrong'; save_state(raw, root / 'teacher.pt')
    before = files(root); counts = dict(toy['calls'])
    choice = dict(toy['candidate'], step=25)
    with pytest.raises(ValueError): search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(2700,), device='cpu')
    assert files(root) == before and toy['calls']['student'] == counts['student']
    assert not (root / 'selected.json').exists()


def test_dispatch_new_teacher_kind(monkeypatch):
    import src.research_loop as loop
    monkeypatch.setattr(teacher, 'prepare_job', lambda **kwargs: kwargs)
    stop = lambda: False
    result = loop.dispatch(dict(kind='citation_gcn_teacher', options={'dataset': 'cora'}), stop)
    assert result == dict(dataset='cora', stop=stop)


@pytest.fixture
def partial(toy):
    """A real synthetic two-epoch GCN fit, distinct from dataset research fits."""
    from src.gcn_teacher import fit_gcn_teacher
    folder = toy['output'] / 'toy_two_epoch'
    fit_gcn_teacher(toy['graph'], toy['train'], toy['validation'], folder, epochs=2, seed=0)
    state = load(folder / 'teacher_resume.pt')
    recipe = teacher._teacher_recipe(toy['graph'], toy['train'], toy['validation'])
    state.update(recipe=recipe, fingerprint=_fingerprint(recipe))
    _, _, _, config = teacher.context(toy['graph'], toy['train'], toy['validation'], toy['h'], toy['config'], toy['source'], toy['pin'])
    root = teacher.teacher_root(toy['output'], config)
    root.mkdir()
    save_json(config, root / 'config.json'); save_json(config['gcn_context'], root / 'source_binding.json')
    import shutil
    shutil.copyfile(toy['source'] / 'propagated_H.pt', root / 'propagated_H.pt')
    shutil.copyfile(toy['source'] / 'inputs_0.pt', root / 'inputs_0.pt')
    teacher._execution(root, toy['graph'], complete=False, create=True)
    (root / 'gcn_teacher').mkdir(); save_state(state, root / 'gcn_teacher' / 'teacher_resume.pt')
    return root, state, config


def test_valid_partial_resume_guard_and_no_partial_condensation(toy, partial):
    root, state, config = partial
    teacher._resume_validate(state, toy['graph'], toy['train'], toy['validation'], config['gcn_context'])
    before = files(root); count = dict(toy['calls'])
    with pytest.raises(ValueError, match='all 200'): screen(toy)
    assert files(root) == before and toy['calls']['optimizer'] == count['optimizer']


@pytest.mark.parametrize('change', ['epoch_bool', 'epoch_float', 'history', 'best', 'best_state', 'model_nan',
                                  'model_dtype', 'Adam_lr', 'Adam_wd', 'Adam_missing', 'Adam_nan', 'Adam_negative',
                                  'Adam_step', 'RNG_torch', 'RNG_python', 'RNG_numpy', 'RNG_cuda', 'timings', 'context'])
def test_partial_resume_malformed_rejects_before_delegate_and_writes(toy, partial, change):
    root, state, config = partial
    if change == 'epoch_bool': state['epoch'] = True
    elif change == 'epoch_float': state['epoch'] = 2.
    elif change == 'history': state['history'].pop()
    elif change == 'best': state['best'] = None
    elif change == 'best_state': next(iter(state['best_state'].values())).view(-1)[0] += 1.
    elif change == 'model_nan': next(iter(state['model_state'].values())).view(-1)[0] = float('nan')
    elif change == 'model_dtype': state['model_state'] = {k: v.double() for k, v in state['model_state'].items()}
    elif change == 'Adam_lr': state['optimizer']['param_groups'][0]['lr'] *= 2
    elif change == 'Adam_wd': state['optimizer']['param_groups'][0]['weight_decay'] = 0
    elif change == 'Adam_missing': del state['optimizer']['state'][0]['exp_avg']
    elif change == 'Adam_nan': state['optimizer']['state'][0]['exp_avg'].view(-1)[0] = float('nan')
    elif change == 'Adam_negative': state['optimizer']['state'][0]['exp_avg_sq'].view(-1)[0] = -1
    elif change == 'Adam_step': state['optimizer']['state'][0]['step'] += 1
    elif change == 'RNG_torch': state['rng']['torch'] = torch.zeros_like(state['rng']['torch'])
    elif change == 'RNG_python': state['rng']['python'] = [0, [], 0]
    elif change == 'RNG_numpy': state['rng']['numpy'] = ['bad']
    elif change == 'RNG_cuda': state['rng']['cuda'] = [torch.tensor([0], dtype=torch.uint8)]
    elif change == 'timings': state['timings']['training_seconds'] = float('inf')
    else: state['recipe']['input_digest'] = 'bad'
    save_state(state, root / 'gcn_teacher' / 'teacher_resume.pt')
    before = files(root); count = dict(toy['calls']); rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError): prepare(toy)
    assert files(root) == before and toy['calls']['teacher'] == count['teacher']
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_partial_resume_device_class_rejected_before_delegate(toy, partial, monkeypatch):
    root, _, _ = partial
    monkeypatch.setattr(teacher, '_device_class', lambda graph: 'cuda')
    before = files(root)
    with pytest.raises(ValueError, match='cross-device'): prepare(toy)
    assert files(root) == before and toy['calls']['teacher'] == 0


def test_completed_fit_device_is_preserved_on_CPU_readonly_audit(toy):
    root = Path(prepare(toy)['root']); execution = json.loads((root / 'teacher_execution.json').read_text())
    execution['fit_device_class'] = 'cuda'; save_json(execution, root / 'teacher_execution.json')
    selected = load(root / 'teacher.pt'); selected['native_fit_execution'] = execution; save_state(selected, root / 'teacher.pt')
    before = files(root)
    teacher.validate_root(root, toy['graph'], toy['train'], toy['validation'], toy['h'])
    assert files(root) == before


@pytest.mark.parametrize('kind', ['embedded_backend', 'row_wrong_root', 'cond_seed', 'derived_dtype'])
def test_selection_and_target_guard_edge_cases_before_bypass(toy, kind):
    root = Path(prepare(toy)['root']); screen(toy); before = files(root); counts = dict(toy['calls'])
    choice = dict(toy['candidate'], step=25)
    if kind == 'embedded_backend': choice['teacher_backend'] = teacher.BACKEND
    elif kind == 'row_wrong_root': choice['candidate_path'] = str(toy['source'] / _fingerprint(toy['candidate']))
    if kind == 'derived_dtype':
        selected = load(root / 'teacher.pt'); q = (selected['logits'] / .3).softmax(1)
        z = load(root / 'inputs_0.pt')['z']; assignment = load(toy['pin']['source_hard_assignment_path'])
        with pytest.raises(ValueError, match='double probabilities'):
            teacher.validate_condensation(root, toy['candidate'], 0, z, q, assignment, 25)
    else:
        with pytest.raises(ValueError): search.selected_test(root, choice, condensation_seeds=(1,) if kind == 'cond_seed' else (0,),
                                                             student_seeds=(2700,), device='cpu')
    assert files(root) == before and toy['calls']['student'] == counts['student'] and not (root / 'selected.json').exists()


def test_stop_preload_does_not_create_teacher_or_run_dataset(toy):
    before = files(toy['output'])
    with pytest.raises(InterruptedError): prepare(toy, stop=lambda: True)
    assert files(toy['output']) == before and toy['calls']['data'] == toy['calls']['teacher'] == 0


@pytest.mark.parametrize('change', ['missing_input_binding', 'input_context', 'input_device', 'native_envelope'])
def test_missing_or_malformed_context_proof_rejects_cached_bypass(toy, change):
    root = Path(prepare(toy)['root']); screen(toy)
    path = root / _fingerprint(toy['candidate']) / 'condensation_0' / 'gcn_input_binding.json'
    if change == 'native_envelope': save_state([], root / 'gcn_teacher' / 'teacher.pt')
    elif change == 'missing_input_binding': path.unlink()
    else:
        state = json.loads(path.read_text()); state['context_digest' if change == 'input_context' else 'factor_device_class'] = 'bad'
        save_json(state, path)
    before = files(root); counts = dict(toy['calls'])
    with pytest.raises(ValueError): screen(toy)
    assert files(root) == before and toy['calls']['optimizer'] == counts['optimizer'] and toy['calls']['student'] == counts['student']


def test_cached_step0_selection_preflight_allows_complete25_state(toy):
    root = Path(prepare(toy)['root']); screen(toy)
    selected = load(root / 'teacher.pt'); q = (selected['logits'].double() / .3).softmax(1)
    teacher.validate_condensation(root, toy['candidate'], 0, load(root / 'inputs_0.pt')['z'], q,
                                  load(toy['pin']['source_hard_assignment_path']), 0)


@pytest.mark.parametrize('case', ['steps', 'checkpoint', 'autocast', 'selected_step'])
def test_fixed_pilot_budget_and_native_precision_preload_guard(toy, case):
    if case == 'selected_step':
        root = Path(prepare(toy)['root']); screen(toy); before = files(root)
        with pytest.raises(ValueError, match='checkpoint'):
            search.selected_test(root, dict(toy['candidate'], step=50), condensation_seeds=(0,), device='cpu')
        assert files(root) == before
        return
    before = files(toy['output'])
    if case == 'autocast':
        with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='autocast'):
            prepare(toy)
    else:
        with pytest.raises(ValueError):
            search.run_screen('cora', .013, toy['output'], [toy['candidate']], teacher_backend=teacher.BACKEND,
                              initialization_source=toy['pin'], steps=50 if case == 'steps' else 25,
                              checkpoints=[2] if case == 'checkpoint' else None, device='cpu')
    assert files(toy['output']) == before and toy['calls']['data'] == 0
