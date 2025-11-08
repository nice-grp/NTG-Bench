from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _resolve_candidate_files(base_path: Path, class_name: str) -> List[Path]:
    candidates = [
        base_path / f"{class_name}_cleaned.csv",
        base_path / f"{class_name}.csv",
    ]
    return [path for path in candidates if path.exists()]


def load_and_label_data(
    base_path: Path | str, classes: Iterable[str], label_col: str
) -> pd.DataFrame:
    """
    Load per-class CSV files, append a label column, and concatenate them.

    The helper mirrors the dataset layout assumed by the paper: each traffic
    class is stored in an individual CSV file and optional `_cleaned` variants
    are preferred when available.
    """

    base = Path(base_path)
    frames: List[pd.DataFrame] = []

    for class_name in classes:
        candidates = _resolve_candidate_files(base, class_name)

        if not candidates:
            LOGGER.warning("Missing data for class '%s' in %s", class_name, base)
            continue

        path = candidates[0]
        LOGGER.debug("Loading %s", path)
        df = pd.read_csv(path)
        df[label_col] = class_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
