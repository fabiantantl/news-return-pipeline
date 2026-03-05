"""Utilities for importing a cleaned prototyping dataset from Kaggle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_CANDIDATES = ("date", "datetime", "timestamp", "time")
HEADLINE_CANDIDATES = ("headline", "title", "news", "text", "article", "summary")
CLOSE_CANDIDATES = ("close", "adj_close", "adj close", "price", "sp500", "index_close")


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def _pick_column(columns: list[str], candidates: tuple[str, ...], semantic_name: str) -> str | None:
    normalized = {_normalize_name(col): col for col in columns}
    for candidate in candidates:
        found = normalized.get(_normalize_name(candidate))
        if found:
            return found
    if semantic_name == "close":
        return None
    raise ValueError(
        f"Could not find a '{semantic_name}' column. Available columns: {columns}"
    )


def normalize_prototype_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Kaggle dataframe into required pipeline schema."""

    columns = list(df.columns)
    date_col = _pick_column(columns, DATE_CANDIDATES, "date")
    headline_col = _pick_column(columns, HEADLINE_CANDIDATES, "headline")
    close_col = _pick_column(columns, CLOSE_CANDIDATES, "close")

    selected_columns = [date_col, headline_col] + ([close_col] if close_col else [])
    cleaned = df.loc[:, selected_columns].copy()
    rename_map = {date_col: "date", headline_col: "headline"}
    if close_col:
        rename_map[close_col] = "close"
    cleaned = cleaned.rename(columns=rename_map)

    if "close" not in cleaned.columns:
        cleaned["close"] = pd.NA

    cleaned = cleaned.loc[:, ["date", "headline", "close"]]
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cleaned["headline"] = cleaned["headline"].astype(str).str.strip()
    cleaned["close"] = pd.to_numeric(cleaned["close"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "headline"]).reset_index(drop=True)

    return cleaned


def resolve_dataset_csv(download_dir: str | Path) -> Path:
    """Pick the best candidate CSV from a KaggleHub dataset directory recursively."""

    root = Path(download_dir)
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {root}")

    scored: list[tuple[int, int, Path, list[str]]] = []
    for path in csv_files:
        columns = list(pd.read_csv(path, nrows=0).columns)
        normalized = {_normalize_name(col) for col in columns}

        has_date = any(_normalize_name(c) in normalized for c in DATE_CANDIDATES)
        has_headline = any(_normalize_name(c) in normalized for c in HEADLINE_CANDIDATES)
        has_close = any(_normalize_name(c) in normalized for c in CLOSE_CANDIDATES)
        score = int(has_date) + int(has_headline) + int(has_close)
        scored.append((score, path.stat().st_size, path, columns))

    eligible = [item for item in scored if item[0] >= 2]
    if not eligible:
        inspected = "\n".join(
            f"- {path.relative_to(root)}: columns={columns}"
            for _, _, path, columns in scored
        )
        raise ValueError(
            "Could not find a suitable CSV with at least date + headline columns. "
            f"Inspected files:\n{inspected}"
        )

    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return eligible[0][2]
