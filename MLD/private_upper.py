"""Private upper bound (UQE) and clipping for the MLD release."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core import (
    _validated_vector,
    exact_smooth_sensitivity_mld,
    mld,
    polynomial_mechanism_parameters,
)


@dataclass(frozen=True)
class UQEDraws:
    """Raw outputs and mechanism diagnostics from repeated UQE draws."""

    private_u: np.ndarray
    log_private_u: np.ndarray
    grid_index: np.ndarray
    used_geometric_tail: np.ndarray
    geometric_wait: np.ndarray
    first_all_below_index: int
    target_count: float


def uqe_candidate_from_index(
    index: int,
    lower_bound: float = 1.0,
    growth: float = 1.01,
) -> tuple[float, float]:
    """Return ``(U_k, log(U_k))`` for ``U_k=L+growth**k-1``."""
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise TypeError("index must be an integer.")
    if index < 0:
        raise ValueError("index must be nonnegative.")
    if not (
        np.isfinite(lower_bound)
        and np.isfinite(growth)
        and lower_bound > 0.0
        and growth > 1.0
    ):
        raise ValueError("Require lower_bound > 0 and growth > 1.")
    log_power = index * math.log(growth)
    if log_power > math.log(np.finfo(float).max):
        raise OverflowError("The sampled UQE grid index exceeds float64 range.")
    candidate = lower_bound + math.exp(log_power) - 1.0
    return candidate, math.log(candidate)


def build_uqe_prefix(
    x_sorted: np.ndarray,
    lower_bound: float = 1.0,
    growth: float = 1.01,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the finite grid prefix before ``count(x_i < U_k)`` is ``n``."""
    values = _validated_vector(x_sorted, minimum_size=1)
    if np.any(np.diff(values) < 0.0):
        raise ValueError("x_sorted must be sorted in nondecreasing order.")
    if not (
        np.isfinite(lower_bound)
        and np.isfinite(growth)
        and lower_bound > 0.0
        and growth > 1.0
    ):
        raise ValueError("Require lower_bound > 0 and growth > 1.")
    if np.any(values < lower_bound):
        raise ValueError("Every observation must be at least lower_bound.")

    shifted_max = float(values[-1] - lower_bound + 1.0)
    first_all = math.floor(math.log(shifted_max) / math.log(growth)) + 1
    while uqe_candidate_from_index(first_all, lower_bound, growth)[0] <= values[-1]:
        first_all += 1
    while (
        first_all > 0
        and uqe_candidate_from_index(first_all - 1, lower_bound, growth)[0]
        > values[-1]
    ):
        first_all -= 1

    indices = np.arange(first_all, dtype=np.float64)
    prefix = lower_bound + np.exp(indices * math.log(growth)) - 1.0
    counts = np.searchsorted(values, prefix, side="left").astype(np.float64)
    return prefix, counts, first_all


def sample_geometric_wait(
    log_success_probability: float,
    rng: np.random.Generator,
) -> int:
    """Sample ``Geometric(p)`` on ``{1,2,...}`` stably from ``log(p)``."""
    if not np.isfinite(log_success_probability) or log_success_probability > 0:
        raise ValueError("log_success_probability must be finite and <= 0.")
    success_probability = math.exp(log_success_probability)
    if success_probability == 0.0:
        raise OverflowError("The geometric success probability underflowed.")
    if success_probability == 1.0:
        return 1
    log_failure = math.log1p(-success_probability)
    log_survival = math.log1p(-float(rng.random()))
    return int(math.floor(log_survival / log_failure) + 1)


def sample_unbounded_uqe(
    x_sorted: np.ndarray,
    epsilon_u: float,
    n_draws: int,
    rng: np.random.Generator,
    lower_bound: float = 1.0,
    growth: float = 1.01,
    threshold_budget_fraction: float = 0.5,
    target_offset: float | None = None,
) -> UQEDraws:
    """Sample UQE upper bounds, with an exact geometric tail beyond the data.

    The target count is ``n + 1/epsilon_u``; ``epsilon_u`` is split between the
    threshold and the queries.
    """
    values = _validated_vector(x_sorted, minimum_size=1)
    if np.any(np.diff(values) < 0.0):
        raise ValueError("x_sorted must be sorted in nondecreasing order.")
    if not np.isfinite(epsilon_u) or epsilon_u <= 0.0:
        raise ValueError("epsilon_u must be positive.")
    if not isinstance(n_draws, (int, np.integer)) or isinstance(n_draws, bool):
        raise TypeError("n_draws must be an integer.")
    if n_draws < 1:
        raise ValueError("n_draws must be positive.")
    if not 0.0 < threshold_budget_fraction < 1.0:
        raise ValueError("threshold_budget_fraction must lie in (0,1).")

    epsilon_threshold = epsilon_u * threshold_budget_fraction
    epsilon_queries = epsilon_u - epsilon_threshold
    offset = 1.0 / epsilon_u if target_offset is None else float(target_offset)
    if not np.isfinite(offset):
        raise ValueError("target_offset must be finite.")
    target = values.size + offset
    prefix, below_counts, first_all = build_uqe_prefix(
        values, lower_bound, growth
    )

    private_u = np.empty(n_draws, dtype=np.float64)
    log_private_u = np.empty(n_draws, dtype=np.float64)
    grid_index = np.empty(n_draws, dtype=np.int64)
    used_tail = np.zeros(n_draws, dtype=bool)
    waits = np.zeros(n_draws, dtype=np.int64)

    for draw in range(n_draws):
        threshold_noise = float(rng.exponential(1.0 / epsilon_threshold))
        noisy_target = target + threshold_noise
        noisy_counts = below_counts + rng.exponential(
            1.0 / epsilon_queries, size=prefix.size
        )
        crossing = np.flatnonzero(noisy_counts >= noisy_target)
        if crossing.size:
            selected = int(crossing[0])
        else:
            log_p = -epsilon_queries * (
                target - values.size + threshold_noise
            )
            wait = sample_geometric_wait(log_p, rng)
            selected = first_all + wait - 1
            used_tail[draw] = True
            waits[draw] = wait

        candidate, log_candidate = uqe_candidate_from_index(
            selected, lower_bound, growth
        )
        private_u[draw] = candidate
        log_private_u[draw] = log_candidate
        grid_index[draw] = selected

    return UQEDraws(
        private_u=private_u,
        log_private_u=log_private_u,
        grid_index=grid_index,
        used_geometric_tail=used_tail,
        geometric_wait=waits,
        first_all_below_index=first_all,
        target_count=float(target),
    )


def release_private_upper_mld(x: np.ndarray, epsilon_u: float, epsilon_mld: float,
                              L: float = 1., growth: float = 1.01,
                              *, rng: np.random.Generator | None = None) -> float:
    """Private upper bound, clipping and Cauchy release; budget epsilon_u + epsilon_mld."""
    values = np.sort(_validated_vector(x))
    beta, alpha = polynomial_mechanism_parameters(epsilon_mld, 2.)
    rng = np.random.default_rng() if rng is None else rng
    upper = float(sample_unbounded_uqe(values, epsilon_u, 1, rng, L, growth).private_u[0])
    if upper == L:
        return 0.
    clipped = np.minimum(values, upper)
    smooth = exact_smooth_sensitivity_mld(clipped, beta, L, upper)
    return float(mld(clipped) + smooth / alpha * rng.standard_cauchy())
