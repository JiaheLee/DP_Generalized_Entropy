"""Ordinary bootstrap of the MLD for the CPS application."""

from __future__ import annotations

import numpy as np


N_BOOTSTRAP = 2_000
BATCH_SIZE = 20
RANDOM_SEED = 123


def bootstrap_mld(
    x: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Return MLDs of ordinary size-n bootstrap samples, drawn in batches."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("x must be a one-dimensional vector of length >= 2.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Bootstrap MLD requires finite positive values.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    n = values.size
    log_values = np.log(values)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)

    completed = 0
    while completed < n_bootstrap:
        current = min(batch_size, n_bootstrap - completed)
        indices = rng.integers(0, n, size=(current, n), dtype=np.int64)
        draws[completed : completed + current] = (
            np.log(np.mean(values[indices], axis=1))
            - np.mean(log_values[indices], axis=1)
        )
        completed += current
        if verbose and (completed % max(100, batch_size) == 0 or completed == n_bootstrap):
            print(f"Bootstrap: {completed:,}/{n_bootstrap:,}", flush=True)

    return draws


def empirical_bootstrap_sd(
    x: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    *,
    verbose: bool = False,
) -> float:
    """Return the empirical bootstrap SD using the sample-SD denominator."""
    draws = bootstrap_mld(
        x,
        n_bootstrap=n_bootstrap,
        batch_size=batch_size,
        seed=seed,
        verbose=verbose,
    )
    return float(np.std(draws, ddof=1))
