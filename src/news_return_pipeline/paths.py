"""Repo-relative path helpers."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent


def data_raw_dir() -> Path:
    """Return repo-relative raw data directory."""

    return REPO_ROOT / "data" / "raw"


def data_processed_dir() -> Path:
    """Return repo-relative processed data directory."""

    return REPO_ROOT / "data" / "processed"
