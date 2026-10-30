# Differentially Private Mean Log Deviation and Theil Index

Code for releasing the mean log deviation (MLD) and the Theil index under
pure ε-differential privacy with smooth sensitivity, and for reproducing the
CPS application and the repeated-sampling study in the paper.

Each release has two stages:

1. **Private upper bound.** An unbounded quantile estimator (AboveThreshold on
   the geometric grid `U_k = L + 1.01^k - 1`, target count `n + 1/ε_U`) selects
   an upper bound `Ũ` with budget `ε_U`. Incomes are clipped to `[L, Ũ]`.
2. **Index release.** The index of the clipped data is released with Cauchy
   noise of scale `S / α`, where `S` is a β-smooth bound on the local
   sensitivity, `β = ε_f / 2` and `α = ε_f / 8`.

The total budget is `ε_U + ε_f`. For the MLD, `S` is the exact smooth
sensitivity. For the Theil index, `S` is the smooth upper bound built from the
neighborhood bounds `B_T^(k)`.

## Repository layout

```text
.
├── MLD/
│   ├── core.py                 # MLD, exact LS/GS/A^(k), smooth sensitivity
│   ├── private_upper.py        # UQE upper bound, clipping
│   ├── application/run.py      # CPS personal and household experiments
│   ├── application/bootstrap.py
│   └── simulation/run.py       # Repeated-sampling study
├── Theil/
│   ├── core.py                 # Theil index, LS/GS, B_T^(k), smooth upper bound
│   ├── private_upper.py        # UQE upper bound in log space, clipping
│   ├── application/run.py
│   ├── application/bootstrap.py
│   └── simulation/run.py
├── common/
│   ├── clean_cps_data.py       # Builds the cleaned personal and household files
│   ├── calibrate.py            # Fits the zero-inflated GB2 population
│   ├── calibration/zero_mass_gb2_parameters.csv
│   ├── data.py                 # Loads the personal and household samples
│   ├── population.py           # GB2 income distribution and reproducible random streams
│   └── reporting.py            # Tables, plot caches and figures
├── Data/                       # CPS ASEC 2025 input files (see below)
├── output/                     # Combined paper figures
├── tests/test_toy_examples.py  # Appendix examples
├── replot.py                   # Redraws figures from saved results only
└── requirements.txt
```

Every experiment directory already contains the results used in the paper
(`main_table.csv`, `appendix_tables.csv`, figures and `plot_data.npz`).

## Installation

Tested with Python 3.10.

```bash
python -m pip install -r requirements.txt
```

## Data

The application uses the 2025 CPS Annual Social and Economic Supplement
public-use files (income year 2024) from the U.S. Census Bureau.

Download the CSV data file from
[2025 CPS ASEC dataset page](https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.2025.html).

The simulations and the tests do not need these files, because the fitted GB2
parameters are included in `common/calibration/`.
