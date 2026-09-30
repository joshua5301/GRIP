import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_table(table, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.csv")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def save_state(state, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(cpu_state(state), temporary)
    temporary.replace(path)


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_state(item) for item in value]
    return value
