"""Shared paths and input validation for the CPS real-data experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_INPUT_CSV = DATA_DIR / "pppub25_clean.csv"
DEFAULT_INCOME_COLUMN = "PTOTVAL"


def load_shifted_income(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    income_column: str = DEFAULT_INCOME_COLUMN,
) -> np.ndarray:
    """Return sorted nonnegative incomes plus one, so the lower bound is L = 1."""
    input_path = Path(input_csv)
    data = pd.read_csv(input_path, low_memory=False)
    data.columns = [str(column).upper() for column in data.columns]
    column = income_column.upper()
    if column not in data.columns:
        raise ValueError(
            f"{column} is not in {input_path}. "
            f"Available columns are {list(data.columns)}"
        )

    income = pd.to_numeric(data[column], errors="coerce")
    income = income[
        income.notna() & np.isfinite(income) & (income >= 0)
    ].to_numpy(dtype=np.float64)
    if income.size < 2:
        raise ValueError("The cleaned data have fewer than two usable incomes.")

    shifted = np.sort(income + 1.0)
    if np.any(shifted < 1.0):
        raise ValueError("Shifted incomes must lie in the public domain x>=1.")
    return shifted


HOUSEHOLD_INPUT_CSV = DATA_DIR / "hhpub25_clean.csv"
HOUSEHOLD_INCOME_COLUMN = "HTOTVAL"


def load_cps(sample: str) -> np.ndarray:
    """Load the same CPS sample for either inequality index.

    Both cleaned files are written by ``python -m common.clean_cps_data``.
    """
    if sample == "personal":
        return load_shifted_income()
    if sample == "household":
        return load_shifted_income(HOUSEHOLD_INPUT_CSV, HOUSEHOLD_INCOME_COLUMN)
    raise ValueError("sample must be personal or household.")
