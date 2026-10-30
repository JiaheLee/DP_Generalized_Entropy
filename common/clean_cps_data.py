"""Build the cleaned CPS ASEC 2025 personal and household files.

Run once after downloading the raw files into Data/:

    python -m common.clean_cps_data
"""

from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


def clean_pppub25(
    input_path: str | Path = DATA_DIR / "pppub25.csv",
    output_path: str | Path = DATA_DIR / "pppub25_clean.csv",
) -> pd.DataFrame:
    """Keep adults (18+) with a positive weight and nonmissing, nonnegative PTOTVAL."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [str(col).upper() for col in df.columns]

    required_cols = [
        "PH_SEQ",
        "PPPOS",
        "A_AGE",
        "MARSUPWT",
        "PTOTVAL",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns include: {list(df.columns[:30])}"
        )

    cleaned = df[required_cols].copy()

    numeric_cols = ["A_AGE", "MARSUPWT", "PTOTVAL"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    mask = (
        (cleaned["A_AGE"] >= 18)
        & (cleaned["MARSUPWT"] > 0)
        & cleaned["PTOTVAL"].notna()
        & (cleaned["PTOTVAL"] >= 0)
    )

    cleaned = cleaned.loc[mask].copy()

    cleaned = cleaned.sort_values(
        by=["PTOTVAL", "PH_SEQ", "PPPOS"],
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Original number of records: {len(df):,}")
    print(f"Cleaned number of records:  {len(cleaned):,}")
    print(f"Minimum income: {cleaned['PTOTVAL'].min():,.2f}")
    print(f"Maximum income: {cleaned['PTOTVAL'].max():,.2f}")
    print(f"Mean income:    {cleaned['PTOTVAL'].mean():,.2f}")

    return cleaned


def clean_hhpub25(
    household_path: str | Path = DATA_DIR / "hhpub25.csv",
    person_path: str | Path = DATA_DIR / "pppub25.csv",
    output_path: str | Path = DATA_DIR / "hhpub25_clean.csv",
) -> pd.DataFrame:
    """Keep interviewed households with members, a householder, a positive weight
    and nonmissing, nonnegative HTOTVAL."""
    household_path = Path(household_path)
    person_path = Path(person_path)
    output_path = Path(output_path)

    required_cols = ["H_SEQ", "H_HHTYPE", "H_NUMPER", "HSUP_WGT", "HTOTVAL"]
    households = pd.read_csv(household_path, usecols=required_cols, low_memory=False)
    persons = pd.read_csv(person_path, usecols=["PH_SEQ", "HHDREL"], low_memory=False)
    for frame in (households, persons):
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    householder_ids = persons.loc[persons["HHDREL"] == 1, "PH_SEQ"]
    if householder_ids.duplicated().any():
        raise ValueError("A household has more than one HHDREL=1 record.")

    mask = (
        (households["H_HHTYPE"] == 1)
        & (households["H_NUMPER"] > 0)
        & (households["HSUP_WGT"] > 0)
        & households["H_SEQ"].isin(householder_ids)
        & households["HTOTVAL"].notna()
        & (households["HTOTVAL"] >= 0)
    )

    cleaned = households.loc[mask, required_cols].copy()
    if cleaned["H_SEQ"].duplicated().any():
        raise ValueError("Expected one record per household.")

    cleaned = cleaned.sort_values(
        by=["HTOTVAL", "H_SEQ"],
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"Input files: {household_path}, {person_path}")
    print(f"Output file: {output_path}")
    print(f"Original number of households: {len(households):,}")
    print(f"Cleaned number of households:  {len(cleaned):,}")
    print(f"Minimum income: {cleaned['HTOTVAL'].min():,.2f}")
    print(f"Maximum income: {cleaned['HTOTVAL'].max():,.2f}")
    print(f"Mean income:    {cleaned['HTOTVAL'].mean():,.2f}")

    return cleaned


if __name__ == "__main__":
    clean_pppub25()
    clean_hhpub25()
