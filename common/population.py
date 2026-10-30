"""Shared calibrated GB2 population and deterministic random-stream keys."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration/zero_mass_gb2_parameters.csv"
# Random-stream identifier per sample size; other sizes use n + 10.
SAMPLE_STREAMS = {10_000: 1, 100_000: 2, 1_000_000: 3, 1_000: 4}


def keyed_rng(seed: int, replication: int, sample_index: int, stage: int,
              epsilon_index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [seed, replication, sample_index, stage, epsilon_index]
    ))


@dataclass(frozen=True)
class GB2Population:
    pi_zero: float
    a: float
    b: float
    p: float
    q: float

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "GB2Population":
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError("The calibration CSV must have exactly one row.")
        row = rows[0]
        model = cls(*(float(row[key]) for key in ("pi_zero", "gb2_a", "gb2_b", "gb2_p", "gb2_q")))
        if not 0 <= model.pi_zero < 1 or any(not math.isfinite(v) or v <= 0
                                             for v in (model.a, model.b, model.p, model.q)):
            raise ValueError("Invalid zero-mass GB2 parameters.")
        if model.q <= 1/model.a:
            raise ValueError("Population Theil needs finite E[X] and E[X log X].")
        return model

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 2:
            raise ValueError("n must be >= 2.")
        income = np.zeros(n)
        positive = np.flatnonzero(rng.random(n) >= self.pi_zero)
        v = rng.beta(self.p, self.q, size=positive.size)
        invalid = (v <= 0) | (v >= 1) | (~np.isfinite(v))
        while np.any(invalid):
            v[invalid] = rng.beta(self.p, self.q, size=int(invalid.sum()))
            invalid = (v <= 0) | (v >= 1) | (~np.isfinite(v))
        income[positive] = np.exp(math.log(self.b)+(np.log(v)-np.log1p(-v))/self.a)
        if np.any(~np.isfinite(income)):
            raise FloatingPointError("Income draw exceeds float64 range.")
        return income+1.

    def moments(self) -> dict[str, float]:
        """Population Theil of X=Y+1, not the unshifted GB2 formula."""
        from scipy.integrate import quad
        from scipy.special import betaln

        log_beta = float(betaln(self.p, self.q))
        mean_y = self.b*math.exp(float(betaln(self.p+1/self.a, self.q-1/self.a))-log_beta)

        def integrand(logit_v: float) -> float:
            log_density = (-self.p*np.logaddexp(0., -logit_v)
                           - self.q*np.logaddexp(0., logit_v)-log_beta)
            log_x = float(np.logaddexp(0., math.log(self.b)+logit_v/self.a))
            return math.exp(float(log_density)+log_x)*log_x

        positive_x_log_x, error = quad(integrand, -np.inf, np.inf,
                                        epsabs=1e-7, epsrel=1e-11, limit=500)
        mean_x = 1+(1-self.pi_zero)*mean_y
        mean_x_log_x = (1-self.pi_zero)*positive_x_log_x
        return {
            "population_theil_shifted": mean_x_log_x/mean_x-math.log(mean_x),
            "mean_shifted_income": mean_x,
            "expected_x_log_x": mean_x_log_x,
            "quadrature_error_x_log_x": (1-self.pi_zero)*error,
            "quadrature_error_theil": (1-self.pi_zero)*error/mean_x,
        }
