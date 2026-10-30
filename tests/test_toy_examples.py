"""Check the implementations against the toy examples in the paper's appendix.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MLD.core import (  # noqa: E402
    exact_Ak_mld,
    exact_local_sensitivity_mld,
    exact_smooth_sensitivity_mld,
    global_sensitivity_mld,
    mld,
)
from Theil.core import (  # noqa: E402
    exact_local_sensitivity_theil,
    global_sensitivity_theil,
    neighborhood_bound_theil,
    smooth_upper_bound_theil,
    theil,
)


X = np.array([2.0, 3.0, 5.0, 8.0, 12.0])
L, U = 1.0, 20.0
EPSILON = 1.0
BETA, ALPHA = EPSILON / 2, EPSILON / 8


class MLDToyExample(unittest.TestCase):
    """Appendix: Example of the MLD."""

    def test_index_and_global_sensitivity(self):
        self.assertAlmostEqual(mld(X), 0.1987, places=4)
        self.assertAlmostEqual(global_sensitivity_mld(X.size, L, U), 0.9695, places=4)

    def test_local_sensitivity(self):
        self.assertAlmostEqual(exact_local_sensitivity_mld(X, L, U), 0.1788, places=4)

    def test_neighborhood_sensitivities(self):
        prefix = np.concatenate(([0.0], np.cumsum(X)))
        expected = (0.1788, 0.3063, 0.5337, 0.7600, 0.9258, 0.9695)
        for k, value in enumerate(expected):
            with self.subTest(k=k):
                self.assertAlmostEqual(exact_Ak_mld(X, prefix, k, L, U), value, places=4)

    def test_smooth_sensitivity_and_noise_scale(self):
        smooth = exact_smooth_sensitivity_mld(X, BETA, L, U)
        self.assertAlmostEqual(smooth, 0.1963, places=4)
        self.assertAlmostEqual(smooth / ALPHA, 1.571, places=3)


class TheilToyExample(unittest.TestCase):
    """Appendix: Example of the Theil Index."""

    def test_index_and_global_sensitivity(self):
        self.assertAlmostEqual(theil(X), 0.1810, places=4)
        self.assertAlmostEqual(global_sensitivity_theil(X.size, L, U), 0.9278, places=4)

    def test_local_sensitivity(self):
        self.assertAlmostEqual(exact_local_sensitivity_theil(X, L, U), 0.1428, places=4)

    def test_neighborhood_bounds(self):
        expected = (0.5355, 0.8280, 0.9278, 0.9278, 0.9278)
        for k, value in enumerate(expected):
            with self.subTest(k=k):
                self.assertAlmostEqual(neighborhood_bound_theil(X, k, L, U), value, places=4)

    def test_smooth_upper_bound_and_noise_scale(self):
        smooth = smooth_upper_bound_theil(X, BETA, L, U)
        self.assertAlmostEqual(smooth, 0.5355, places=4)
        self.assertAlmostEqual(smooth / ALPHA, 4.284, places=3)


class BruteForceChecks(unittest.TestCase):
    """Compare closed forms with direct searches on small datasets."""

    def test_mld_local_sensitivity_matches_grid_search(self):
        rng = np.random.default_rng(0)
        grid = np.linspace(L, U, 2001)
        for _ in range(20):
            x = rng.uniform(L, U, size=int(rng.integers(2, 7)))
            brute = 0.0
            for m in range(x.size):
                for y in grid:
                    changed = x.copy()
                    changed[m] = y
                    brute = max(brute, abs(mld(changed) - mld(x)))
            # The exact value can only exceed a grid search, and only slightly.
            exact = exact_local_sensitivity_mld(x, L, U)
            self.assertGreaterEqual(exact, brute - 1e-12)
            self.assertLess(exact - brute, 1e-4)

    def test_theil_neighborhood_bound_dominates_local_sensitivity(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = rng.uniform(L, U, size=int(rng.integers(2, 8)))
            local = exact_local_sensitivity_theil(x, L, U)
            self.assertGreaterEqual(neighborhood_bound_theil(x, 0, L, U), local - 1e-12)

    def test_global_sensitivity_pairs(self):
        for n in (2, 3, 5, 50):
            low = np.full(n, L)
            high = low.copy()
            high[0] = U
            with self.subTest(n=n):
                self.assertAlmostEqual(
                    global_sensitivity_mld(n, L, U), abs(mld(high) - mld(low)), places=12)
                self.assertAlmostEqual(
                    global_sensitivity_theil(n, L, U), abs(theil(high) - theil(low)), places=12)


if __name__ == "__main__":
    unittest.main()
