import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def select_accuracy(rows):
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame[["gamma", "val_acc", "val_ce"]].to_numpy()).all():
        raise ValueError("Teacher scores must be finite")
    return frame.sort_values(["val_acc", "val_ce", "gamma"], ascending=[False, True, True]).iloc[0].to_dict()


def calibrate_temperature(logits, labels, bounds=(0.05, 20.0)):
    low, high = bounds
    if not 0 < low < high or not np.isfinite([low, high]).all():
        raise ValueError("Require finite positive temperature bounds")
    z = logits.detach().double().cpu().numpy()
    y = labels.detach().cpu().numpy()
    if not np.isfinite(z).all():
        raise ValueError("Teacher logits must be finite")
    z = z - z.max(1, keepdims=True)

    def ce(inverse_temperature):
        scaled = z * inverse_temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))

    result = minimize_scalar(ce, bounds=(1 / high, 1 / low), method="bounded", options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError("Temperature calibration failed")
    candidates = [1 / high, 1 / low, float(result.x)]
    if low <= 1 <= high:
        candidates.append(1.0)
    inverse = min(candidates, key=ce)
    temperature = 1 / inverse
    info = dict(
        T=temperature,
        calibrated_val_ce=ce(inverse),
        uncalibrated_val_ce=ce(1.0),
        temperature_boundary="lower"
        if inverse == 1 / low
        else "upper"
        if inverse == 1 / high
        else "interior",
    )
    curve = [
        dict(T=float(t), val_ce=ce(1 / t))
        for t in sorted(set(np.geomspace(low, high, 61).tolist() + [temperature, 1.0]))
    ]
    return info, curve
