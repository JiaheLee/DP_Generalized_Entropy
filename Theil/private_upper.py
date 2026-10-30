"""Private upper bound (UQE kept in log space) and clipping for the Theil release."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core import _vector, release_theil


@dataclass(frozen=True)
class UpperDraws:
    indices: tuple[int, ...]
    log_upper: np.ndarray


def log_grid_value(index: int, L: float = 1., growth: float = 1.01) -> float:
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a nonnegative integer.")
    if not np.isfinite(L) or L <= 0 or not np.isfinite(growth) or growth <= 1:
        raise ValueError("Require finite L>0 and growth>1.")
    if index == 0:
        return math.log(L)
    exponent = index*math.log(growth)
    if not math.isfinite(exponent):
        raise OverflowError("The grid index exceeds even the log representation.")
    if L == 1.:
        return exponent
    log_increment = exponent + math.log(-math.expm1(-exponent))
    return float(np.logaddexp(math.log(L), log_increment))


def _grid_value(index: int, L: float, growth: float) -> float:
    if index == 0:
        return L
    if index*math.log(growth) > math.log(np.finfo(float).max):
        return math.inf  # Only used to locate saturation, never as a release.
    return L + math.expm1(index*math.log(growth))


def build_prefix(x_sorted: np.ndarray, L: float = 1., growth: float = 1.01):
    x = _vector(x_sorted)
    log_grid_value(0, L, growth)  # public-parameter validation
    if np.any(np.diff(x) < 0) or x[0] < L:
        raise ValueError("x must be sorted, with every observation >= L.")
    shifted_max = float(x[-1]-L+1.)
    first_all = math.floor(math.log(shifted_max)/math.log(growth))+1
    while _grid_value(first_all, L, growth) <= x[-1]:
        first_all += 1
    while first_all > 0 and _grid_value(first_all-1, L, growth) > x[-1]:
        first_all -= 1
    candidates = np.array([_grid_value(k, L, growth) for k in range(first_all)])
    counts = np.searchsorted(x, candidates, side="left")
    return counts, first_all


def sample_upper_bounds(x_sorted: np.ndarray, epsilon_u: float, n_draws: int,
                        rng: np.random.Generator, L: float = 1., growth: float = 1.01) -> UpperDraws:
    if not np.isfinite(epsilon_u) or epsilon_u <= 0 or epsilon_u/2 == 0:
        raise ValueError("epsilon_u must be positive and representable after splitting.")
    if not isinstance(n_draws, (int, np.integer)) or isinstance(n_draws, bool) or n_draws < 1:
        raise ValueError("n_draws must be a positive integer.")
    counts, first_all = build_prefix(x_sorted, L, growth)
    deficit = len(x_sorted)-counts
    indices, logs = [], []
    for _ in range(n_draws):
        # Scaled by epsilon_u/2: the noisy target n + 1/epsilon_u becomes n + 1/2 + Exp(1).
        excess = 0.5 + float(rng.exponential())
        queries = rng.exponential(size=counts.size)
        crossing = np.flatnonzero(queries >= deficit*(epsilon_u/2)+excess)
        if crossing.size:
            selected = int(crossing[0])
        else:
            p = math.exp(-excess)
            if p == 0:
                raise OverflowError("Geometric probability exceeds floating-point precision.")
            log_failure = math.log1p(-p)
            wait = math.floor(math.log1p(-float(rng.random()))/log_failure)+1
            selected = first_all+wait-1
        indices.append(selected)
        logs.append(log_grid_value(selected, L, growth))
    return UpperDraws(tuple(indices), np.array(logs))


def clip_at_log_upper(x: np.ndarray, log_upper: float, L: float = 1.) -> np.ndarray:
    x = _vector(x)
    if not np.isfinite(log_upper) or log_upper < math.log(L) or np.any(x < L):
        raise ValueError("Invalid log upper bound or data below L.")
    if log_upper >= math.log(float(np.max(x))):
        return x.copy()
    if log_upper == math.log(L):
        return np.full_like(x, L)
    upper = math.exp(log_upper)
    # Keep rounding in exp() from pushing log(upper) above log_upper.
    while float(np.log(upper)) > log_upper:
        upper = float(np.nextafter(upper, 0.))
    return np.minimum(x, upper)


def release_private_upper_theil(x: np.ndarray, epsilon_u: float, epsilon_theil: float,
                                L: float = 1., growth: float = 1.01,
                                *, rng: np.random.Generator | None = None) -> float:
    """Private upper bound, clipping and Cauchy release; budget epsilon_u + epsilon_theil."""
    x = np.sort(_vector(x))
    rng = np.random.default_rng() if rng is None else rng
    upper = sample_upper_bounds(x, epsilon_u, 1, rng, L, growth).log_upper[0]
    clipped = clip_at_log_upper(x, float(upper), L)
    return release_theil(clipped, epsilon_theil, L, log_upper=float(upper), rng=rng)
