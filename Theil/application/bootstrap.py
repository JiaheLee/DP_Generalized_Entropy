"""Ordinary bootstrap for the Theil real-data experiments."""

import numpy as np

from Theil.core import _positive_remainder


BOOTSTRAP_BATCH_SIZE = 20


def bootstrap_theil(x: np.ndarray, n_bootstrap: int = 2_000, seed: int = 123,
                    batch_size: int = BOOTSTRAP_BATCH_SIZE) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("Invalid bootstrap data.")
    if n_bootstrap < 2 or batch_size < 1:
        raise ValueError("Require n_bootstrap >= 2 and batch_size >= 1.")
    rng = np.random.default_rng(seed)
    scaled = x/x.max()
    values = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, batch_size):
        count = min(batch_size, n_bootstrap-start)
        indices = rng.integers(0, len(x), size=(count, len(x)), dtype=np.int64)
        samples = scaled[indices]
        relative = samples/samples.mean(axis=1, keepdims=True)
        values[start:start+count] = np.mean(_positive_remainder(relative-1), axis=1)
    return values


