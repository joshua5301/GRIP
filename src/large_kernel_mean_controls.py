"""A new cold linear control using exact frozen BW native initial buffers.

Only factory input provision changes: original solver, objective, implicit
cotangent, factor arithmetic and Adam run unchanged. No historical optimizer
endpoint or factory result is recomputed or treated as new qualification.
"""
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path


def _sha(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            check()
            digest.update(block)
    return digest.hexdigest()


def linear25(protocol, stop=lambda: False):
    import torch
    from src import soft_ce_partition as original
    from src.shared_features import _tensor_identity
    from src.sweep_utils import representative
    from types import SimpleNamespace

    started = time.monotonic()
    folder = Path(protocol['linear_control_folder'])
    if folder.exists():
        raise ValueError('A fresh cold matched-control namespace is required')
    folder.mkdir()
    report = dict(schema=1, passed=False, test_enabled=False, student_fits=0,
        operation='NEW cold exact BW initial linear uniform CE fixed25',
        prior_P_updates=0, planned_P_updates=25, condensation_seed=0,
        candidate_condensation_source=protocol['candidate_source'],
        actual_source=protocol['source'], goal_complete=False)
    counters = dict(frozen_initial_copy_attempts=0, frozen_initial_copy_completed=0)
    report['counts'] = counters
    def check():
        if stop() or time.monotonic() - started > 290:
            raise RuntimeError('Terminal bounded cold linear control; preserve incomplete state')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if rss > 16 * 1024**3:
            raise RuntimeError('Cold control RSS16GiB exceeded')
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            if (torch.cuda.max_memory_allocated() > 4 * 1024**3
                    or torch.cuda.max_memory_reserved() > 6 * 1024**3):
                raise RuntimeError('Cold control GPU4/6GiB exceeded')
    def bounded_stop():
        check()
        return False
    def pin_all():
        for path, expected in protocol['immutable_files_sha256'].items():
            if _sha(path, check) != expected:
                raise ValueError('Cold-control immutable input changed: ' + path)
    restored = None
    try:
        pin_all()
        device = torch.device(protocol['device'])
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
        packet = torch.load(protocol['origin']['path'], map_location='cpu', weights_only=True)
        arrays = packet['arrays']
        z, q, hard = [arrays[k].detach().to(device).clone() for k in ('z', 'Q', 'hard')]
        expected = [_tensor_identity(arrays[k]) for k in ('U0', 'V0')]
        initial = [arrays[k].detach().to(device).clone() for k in ('U0', 'V0')]
        original_options = dict(protocol['original_options'])
        original_options.pop('data_digest')
        original_factory = original.initialize_factors
        def frozen_factory(assignment, cells, rank, seed=0):
            counters['frozen_initial_copy_attempts'] += 1
            if (assignment is not hard or cells != 90 or rank != 16 or seed != 0
                    or counters['frozen_initial_copy_attempts'] != 1):
                raise ValueError('Unexpected native factor factory request')
            parameters = [p.detach().clone().requires_grad_() for p in initial]
            if [_tensor_identity(p.detach().cpu()) for p in parameters] != expected:
                raise ValueError('Owning frozen initial U/V copy differs')
            counters['frozen_initial_copy_completed'] += 1
            return parameters
        original.initialize_factors = frozen_factory
        restored = original_factory
        final = original.optimize_ce_assignment(z, q, hard, steps=25,
            folder=folder, checkpoint_steps=(0, 25), save_resume=True,
            stop=bounded_stop, **original_options)
        history = final['history']
        if [row['step'] for row in history] != list(range(26)):
            raise ValueError('Cold linear control lost its complete fixed25 history')
        state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
        if (state['step'] != 25 or state['config'] != protocol['original_options']
                or not all(float(s['step']) == 25 for s in state['optimizer']['state'].values())
                or not state['snapshots'][25]['J_exact']):
            raise ValueError('Cold linear exact25/Adam/config completion changed')
        initial_saved = state['snapshots'][0]['moments']
        if _tensor_identity(initial_saved) != _tensor_identity(arrays['M0']):
            raise ValueError('Cold linear physical M0 differs from original BW')
        transform = SimpleNamespace(**{k:v.to(device) if torch.is_tensor(v) else v
            for k,v in arrays['transform'].items()})
        cx, cy, mass = representative(initial_saved, transform, 128, device)
        readout = [cx, cy, torch.full_like(mass, 1/90)]
        if [_tensor_identity(x.detach().cpu()) for x in readout] != packet['descriptors']['readout']:
            raise ValueError('Cold linear initial physical FP32 readout/uniform weights changed')
        if not all(row['J_exact'] and row['inner_converged'] for row in history):
            raise ValueError('Cold linear original stationary interface failed')
        if not all(row['cg_converged'] for row in history[:25]):
            raise ValueError('Cold linear original update adjoint failed')
        counters.update(completed_history_head_interfaces=26,
            completed_history_update_adjoints=25, completed_optimizer_Adam_steps=25)
        pin_all(); check()
        report.update(passed=True, actual_P_updates=25, continuous_condensations=1,
            complete_history_rows=26, initial_buffers_reused=True,
            initial_physical_readout_preserved=True, new_RNG_draws=0,
            old_condensation_or_proof_replays=0,
            endpoint25={'path':str(folder/'checkpoints/step_000025.pt'),
                'sha256':_sha(folder/'checkpoints/step_000025.pt', check)},
            endpoint0={'path':str(folder/'checkpoints/step_000000.pt'),
                'sha256':_sha(folder/'checkpoints/step_000000.pt', check)},
            physical_moment_identity25=_tensor_identity(state['snapshots'][25]['moments']),
            physical_moment_identity0=_tensor_identity(initial_saved),
            resume_sha256=_sha(folder/'resume.pt', check),
            endpoint_teacher_CE=state['snapshots'][25]['teacher_ce'])
        return report
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error),
            incomplete_operation_interiors='unknown; no automatic retry or rescue')
        raise
    finally:
        if restored is not None:
            original.initialize_factors = restored
        report.update(elapsed_seconds=time.monotonic()-started,
            memory={'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'GPU_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                'GPU_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0})
        with (folder/'control_report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())


def run(protocol_path, protocol_sha256, stop=lambda: False):
    """Terminal cold-control job on an immutable current-source protocol."""
    import torch
    from src.kernel_mean_ce import _runtime
    if _sha(protocol_path) != protocol_sha256:
        raise ValueError('Frozen matched-control protocol changed')
    protocol = json.loads(Path(protocol_path).read_text())
    if (protocol['schema'] != 1 or protocol['test_enabled'] is not False
            or protocol['steps'] != 25 or protocol['operation'] != 'cold_linear25'
            or not callable(stop)):
        raise ValueError('Only frozen validation-only cold linear25 is admitted')
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if (_runtime(torch.device(protocol['device'])) != protocol['runtime']
            or any(os.environ.get(k) != v for k,v in protocol['environment'].items())
            or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
                != protocol['source']['git_head']):
        raise ValueError('Frozen cold-control source/runtime/environment changed')
    try:
        return linear25(protocol, stop)
    except InterruptedError as error:
        raise RuntimeError('Terminal cold linear25 incomplete; no automatic retry') from error
