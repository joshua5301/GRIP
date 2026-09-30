import itertools

import pandas as pd

from src.moments import decode_moments


def grid_rows(space):
    if not space or any(not isinstance(v, (list, tuple)) or not v for v in space.values()):
        raise ValueError("Require nonempty grid lists")
    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def boundary_profile(table, space):
    rows = []
    for key, values in space.items():
        eligible = table[table.step > 0] if key in ("penalty", "assignment_lr") else table
        low, high = min(values), max(values)
        for value, group in eligible.groupby(key):
            rows.append(
                dict(
                    parameter=key,
                    value=value,
                    best_val=group.val.max(),
                    boundary="fixed"
                    if low == high
                    else "lower"
                    if value == low
                    else "upper"
                    if value == high
                    else "interior",
                )
            )
    return pd.DataFrame(rows)


def representative(moments, transform, dimension, device):
    centers, labels, mass = decode_moments(moments.to(device), dimension)
    features = centers * transform.scale + transform.output_center + transform.center
    return features.float(), labels.float(), mass
