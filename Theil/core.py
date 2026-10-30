"""Differentially private Theil index.

Local and global sensitivity, the neighborhood bounds B_T^(k) (rectangle
relaxation capped at GS) and the smooth upper bound on [L, U]^n. Sums are
kept in log space, so a private upper bound can be passed as ``log_upper``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("x must be a finite positive vector of length >= 2.")
    return x


def _bounds(L: float, U: float | None, log_upper: float | None) -> tuple[float, float]:
    if not np.isfinite(L) or L <= 0 or (U is None) == (log_upper is None):
        raise ValueError("Require L > 0 and exactly one of U or log_upper.")
    if U is not None:
        if not np.isfinite(U) or U < L:
            raise ValueError("Require finite U >= L.")
        lu = math.log(U)
    else:
        lu = float(log_upper)
    ll = math.log(L)
    if not np.isfinite(lu) or lu < ll:
        raise ValueError("Require finite log_upper >= log(L).")
    return ll, lu


def _check_domain(x: np.ndarray, L: float, lu: float) -> None:
    # Scalar exp/log and NumPy's vector log can differ by one final ulp.
    if np.any(x < L) or float(np.max(np.log(x))) > np.nextafter(lu, math.inf):
        raise ValueError("Observations must be clipped to [L,U] first.")


def _positive_remainder(d: np.ndarray) -> np.ndarray:
    """(1+d)log(1+d)-d, evaluated without near-zero cancellation."""
    d = np.asarray(d, dtype=float)
    out = np.empty_like(d)
    small = np.abs(d) < 1e-3
    v = d[small]
    out[small] = v * v * (0.5 + v * (-1/6 + v * (1/12 + v * (-1/20 + v/30))))
    v = d[~small]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[~small] = np.where(v == -1, 1., (1 + v) * np.log1p(v) - v)
    return out


def theil(x: np.ndarray) -> float:
    """Unweighted Theil T; stable at equality and for large finite incomes."""
    x = _vector(x)
    scaled = x / np.max(x)
    relative = scaled / np.mean(scaled)
    # Equals mean(r log r) because mean(r - 1) = 0; stays nonnegative near equality.
    return float(np.mean(_positive_remainder(relative - 1)))


def _log_prefix(log_values: np.ndarray) -> np.ndarray:
    return np.r_[-np.inf, np.logaddexp.accumulate(log_values)]


def _signed_add(s1, l1, s2, l2):
    """Add signed quantities represented by (sign, log(abs(value)))."""
    s1, l1, s2, l2 = np.broadcast_arrays(s1, l1, s2, l2)
    pos = np.logaddexp(np.where(s1 > 0, l1, -np.inf), np.where(s2 > 0, l2, -np.inf))
    neg = np.logaddexp(np.where(s1 < 0, l1, -np.inf), np.where(s2 < 0, l2, -np.inf))
    equal = pos == neg
    with np.errstate(invalid="ignore", divide="ignore"):
        sign = np.sign(pos - neg)
        magnitude = np.maximum(pos, neg) + np.log(-np.expm1(-np.abs(pos - neg)))
    return np.where(equal, 0., sign), np.where(equal, -np.inf, magnitude)


def _signed_prefix(sign: np.ndarray, logabs: np.ndarray):
    pos = _log_prefix(np.where(sign > 0, logabs, -np.inf))
    neg = _log_prefix(np.where(sign < 0, logabs, -np.inf))
    return _signed_add(1., pos, -1., neg)


def _phi(logx):
    logx = np.asarray(logx, dtype=float)
    with np.errstate(divide="ignore"):
        return np.sign(logx), logx + np.log(np.abs(logx))


def _phi_extrema(ll: float, lu: float):
    logs = [ll, lu] + ([-1.] if ll <= -1 <= lu else [])
    pairs = [tuple(float(v) for v in _phi(t)) for t in logs]
    pairs.sort(key=lambda z: (z[0], z[1] if z[0] > 0 else -z[1] if z[0] < 0 else 0.))
    return pairs[0], pairs[-1]


def endpoint_function(log_s, log_c, log_t):
    """f_{T,c}(S,t) = c/(S+c) log(c/t) + log((S+t)/(S+c)) from log arguments.

    A series is used near t = c to avoid cancellation.
    """
    ls, lc, lt = np.broadcast_arrays(log_s, log_c, log_t)
    a = lt - lc
    logw = -np.logaddexp(0., ls - lc)
    logv = -np.logaddexp(0., lc - ls)
    # F(w,a) = F(1-w,-a); reduce to a >= 0.
    lp = np.where(a >= 0, logw, logv)
    lq = np.where(a >= 0, logv, logw)
    a = np.abs(a)
    p, q = np.exp(lp), np.exp(lq)
    result = np.empty_like(a)
    small = a < 1e-3
    aa, pp, qq = a[small], p[small], q[small]
    pq = pp * qq
    result[small] = pq * aa**2 * (
        0.5 + (qq-pp)*aa/6 + (1-6*pq)*aa**2/24
        + (qq-pp)*(1-12*pq)*aa**3/120
        + (1-30*pq+120*pq*pq)*aa**4/720
    )
    low = (~small) & (lp <= -math.log(2))
    aa = a[low]
    log_expm1 = aa + np.log(-np.expm1(-aa))
    result[low] = np.logaddexp(0., lp[low] + log_expm1) - p[low]*aa
    high = (~small) & (~low)
    aa, qq = a[high], q[high]
    result[high] = qq*aa + np.log1p(qq*np.expm1(-aa))
    return np.maximum(result, 0.)


def _log_critical_ratio(a: float) -> float:
    """log((r log r-r+1)/(r-1-log r)), r=exp(a), a>=0."""
    if a < 1e-3:
        denominator = sum(a**(j-2)/math.factorial(j) for j in range(2, 10))
        difference = sum((j-2)*a**(j-2)/math.factorial(j) for j in range(3, 10))
        return math.log1p(difference/denominator)
    if a < 1:
        numerator = a*math.exp(a) - math.expm1(a)
        denominator = math.expm1(a) - a
        return math.log(numerator/denominator)
    decay = math.exp(-a)
    return math.log(a-1+decay) - math.log1p(-(1+a)*decay)


def endpoint_maximum(log_s_lower: float, log_s_upper: float, log_c: float, log_t: float) -> float:
    """Closed-form constrained maximum in the manuscript's proposition."""
    a = log_t - log_c
    if a == 0:
        return 0.
    log_star = log_c + math.copysign(_log_critical_ratio(abs(a)), a)
    maximizing_s = min(max(log_star, log_s_lower), log_s_upper)
    return float(endpoint_function(maximizing_s, log_c, log_t))


@dataclass
class PreparedTheil:
    """Sorted log-space summaries used by the neighborhood bounds."""
    n: int
    minimum: float
    log_maximum: float
    log_s_prefix: np.ndarray
    log_s_suffix: np.ndarray
    h_prefix_sign: np.ndarray
    h_prefix_logabs: np.ndarray
    h_suffix_sign: np.ndarray
    h_suffix_logabs: np.ndarray


def prepare_theil(x: np.ndarray) -> PreparedTheil:
    x = np.sort(_vector(x))
    logs = np.log(x)
    signs, magnitudes = _phi(logs)
    keys = np.where(signs < 0, -magnitudes, magnitudes)
    order = np.lexsort((keys, signs))
    signs, magnitudes = signs[order], magnitudes[order]
    ps, pl = _signed_prefix(signs, magnitudes)
    ss, sl = _signed_prefix(signs[::-1], magnitudes[::-1])
    return PreparedTheil(
        x.size, float(x[0]), float(logs[-1]), _log_prefix(logs),
        _log_prefix(logs[::-1])[::-1], ps, pl, ss[::-1], sl[::-1],
    )


def _rectangle(prepared: PreparedTheil | None, n: int, q: int, ll: float, lu: float):
    logq = math.log(q) if q else -math.inf
    if q == n-1:
        small_s = large_s = -math.inf
        hls = hus = 0.
        hll = hul = -math.inf
    else:
        m = n-q-1
        small_s, large_s = prepared.log_s_prefix[m], prepared.log_s_suffix[q+1]
        hls, hll = prepared.h_prefix_sign[m], prepared.h_prefix_logabs[m]
        hus, hul = prepared.h_suffix_sign[q+1], prepared.h_suffix_logabs[q+1]
    low_phi, high_phi = _phi_extrema(ll, lu)
    lower_h = _signed_add(hls, hll, low_phi[0], logq+low_phi[1])
    upper_h = _signed_add(hus, hul, high_phi[0], logq+high_phi[1])
    return (float(np.logaddexp(small_s, logq+ll)), float(np.logaddexp(large_s, logq+lu)),
            lower_h, upper_h)


def _bound_from_rectangle(rectangle, ll: float, lu: float) -> float:
    sl, su, hl, hu = rectangle
    # H/S is log(t*), not t*. Clamp in this space BEFORE exponentiating.
    limit = math.log(max(abs(ll), abs(lu), 1.)) + 1
    ratios = [float(sign)*math.exp(min(float(logabs)-s, limit))
              for sign, logabs in (hl, hu) for s in (sl, su)]
    tl = max(ll, min(lu, min(ratios)))
    tu = max(ll, min(lu, max(ratios)))
    return max(endpoint_maximum(sl, su, ll, tu), endpoint_maximum(sl, su, lu, tl))


def global_sensitivity_theil(n: int, L: float, U: float | None = None,
                             *, log_upper: float | None = None) -> float:
    """Exact GS, attained by (L,...,L) and (U,L,...,L), without exponentiating log_upper."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2.")
    ll, lu = _bounds(L, U, log_upper)
    log_ratio = lu - ll
    if U is not None and U - L <= L:
        # Preserve a small relative width when the finite endpoints are close.
        log_ratio = math.log1p((U - L) / L)
    if log_ratio == 0:
        return 0.
    log_others = math.log(n - 1)
    if log_ratio <= log_others + 1:
        # h(U)-h(L) with S=(n-1)L; the series form avoids cancellation.
        return float(endpoint_function(log_others, log_ratio, 0.))

    # Large U/L: factor U out to avoid subtracting two huge log(U) terms.
    ratio = math.exp(log_others - log_ratio)
    return math.log(n) - math.log1p(ratio) - log_ratio * ratio / (1 + ratio)


def neighborhood_bound_theil(x: np.ndarray | PreparedTheil, k: int, L: float,
                             U: float | None = None, *, log_upper: float | None = None) -> float:
    """B_T^(k) = min(GS, rectangle bound), with B_T^(k) = GS for k>=n-1."""
    ll, lu = _bounds(L, U, log_upper)
    p = x if isinstance(x, PreparedTheil) else prepare_theil(x)
    if p.minimum < L or p.log_maximum > np.nextafter(lu, math.inf):
        raise ValueError("Observations must lie in [L,U].")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a nonnegative integer.")
    if ll == lu:
        return 0.
    global_sensitivity = global_sensitivity_theil(p.n, L, U, log_upper=log_upper)
    if k >= p.n - 1:
        return global_sensitivity
    raw_bound = _bound_from_rectangle(_rectangle(p, p.n, k, ll, lu), ll, lu)
    return min(global_sensitivity, raw_bound)


def global_sensitivity_upper_bound_theil(n: int, L: float, U: float | None = None,
                                         *, log_upper: float | None = None) -> float:
    """Alias of :func:`global_sensitivity_theil`."""
    return global_sensitivity_theil(n, L, U, log_upper=log_upper)


def smooth_upper_bounds_theil(x: np.ndarray | PreparedTheil, betas: np.ndarray, L: float,
                              U: float | None = None, *, log_upper: float | None = None,
                              prune: bool = True) -> tuple[np.ndarray, float]:
    """Return ``(max_k exp(-beta*k) B_T^(k) for each beta, GS)``."""
    betas = np.asarray(betas, dtype=float)
    if betas.ndim != 1 or not betas.size or np.any(~np.isfinite(betas)) or np.any(betas <= 0):
        raise ValueError("betas must be a positive finite vector.")
    p = x if isinstance(x, PreparedTheil) else prepare_theil(x)
    ll, lu = _bounds(L, U, log_upper)
    first = neighborhood_bound_theil(p, 0, L, U, log_upper=log_upper)
    global_sensitivity = global_sensitivity_theil(p.n, L, U, log_upper=log_upper)
    best = np.full(betas.size, first)
    active = np.ones(betas.size, dtype=bool)
    for k in range(1, p.n):
        weights = np.exp(-betas*k)
        if prune:
            active &= weights*global_sensitivity > best
            if not np.any(active):
                break
        bound = (global_sensitivity if k == p.n - 1 else
                 min(global_sensitivity,
                     _bound_from_rectangle(_rectangle(p, p.n, k, ll, lu), ll, lu)))
        best[active] = np.maximum(best[active], weights[active]*bound)
    return best, global_sensitivity


def smooth_upper_bound_theil(x: np.ndarray, beta: float, L: float, U: float | None = None,
                             *, log_upper: float | None = None, prune: bool = True) -> float:
    return float(smooth_upper_bounds_theil(x, np.array([beta]), L, U,
                                         log_upper=log_upper, prune=prune)[0][0])


def exact_local_sensitivity_theil(x: np.ndarray, L: float, U: float | None = None,
                                 *, log_upper: float | None = None) -> float:
    """Exact max_m D_T(S_m,H_m,x_m); O(n), without sorting."""
    x = _vector(x)
    ll, lu = _bounds(L, U, log_upper)
    _check_domain(x, L, lu)
    if ll == lu:
        return 0.
    logs = np.log(x)
    prefix, suffix = _log_prefix(logs), _log_prefix(logs[::-1])[::-1]
    ls = np.logaddexp(prefix[:-1], suffix[1:])
    signs, magnitudes = _phi(logs)
    ps, pl = _signed_prefix(signs, magnitudes)
    ss, sl = _signed_prefix(signs[::-1], magnitudes[::-1])
    hs, hl = _signed_add(ps[:-1], pl[:-1], ss[::-1][1:], sl[::-1][1:])
    lt = np.clip(hs*np.exp(hl-ls), ll, lu)
    # f_x = h(x)-h(t*); endpoint differences are f_L-f_x and f_U-f_x.
    fx = endpoint_function(ls, logs, lt)
    fl, fu = endpoint_function(ls, ll, lt), endpoint_function(ls, lu, lt)
    return float(np.max(np.maximum.reduce([np.abs(fl-fx), np.abs(fu-fx), fx])))


def release_theil(x: np.ndarray, epsilon: float, L: float, U: float | None = None,
                  *, log_upper: float | None = None, rng: np.random.Generator | None = None) -> float:
    """One epsilon-DP Cauchy release of the Theil index on a known domain."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    rng = np.random.default_rng() if rng is None else rng
    scale = smooth_upper_bound_theil(x, epsilon/2, L, U, log_upper=log_upper)/(epsilon/8)
    return float(theil(x) + scale*rng.standard_cauchy())
