"""Repo-relative path helpers."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"


def data_raw_dir() -> Path:
    """Return repo-relative raw data directory."""

    return RAW_DATA_DIR


def data_processed_dir() -> Path:
    """Return repo-relative processed data directory."""

    return PROCESSED_DATA_DIR


def get_raw_path(filename: str) -> Path:
    """Return path to a raw data file under data/raw."""

    return RAW_DATA_DIR / filename


def get_processed_path(filename: str) -> Path:
    """Return path to a processed data file under data/processed."""

    return PROCESSED_DATA_DIR / filename
