"""Raw data loading and schema validation."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("date", "headline", "close")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a raw CSV.

    The CSV must include parseable ``date``, ``headline``, and numeric ``close`` columns.
    Extra columns are ignored.
    """

    input_path = Path(path)
    df = pd.read_csv(input_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Raw CSV is missing required columns: "
            f"{missing}. Required columns are: {list(REQUIRED_COLUMNS)}"
        )

    selected = df.loc[:, list(REQUIRED_COLUMNS)].copy()

    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    if selected["date"].isna().any():
        invalid_count = int(selected["date"].isna().sum())
        raise ValueError(
            f"Column 'date' contains {invalid_count} non-parseable value(s)."
        )

    selected["headline"] = selected["headline"].astype(str)
    selected["close"] = pd.to_numeric(selected["close"], errors="coerce")
    if selected["close"].isna().any():
        invalid_count = int(selected["close"].isna().sum())
        raise ValueError(
            f"Column 'close' contains {invalid_count} non-numeric value(s)."
        )

    return selected
