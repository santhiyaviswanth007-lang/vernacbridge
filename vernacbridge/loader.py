"""
Dataset discovery and raw loading.

The :class:`DatasetLoader` is deliberately "dumb": its only job is finding
CSV files on disk and turning each into a :class:`pandas.DataFrame`, tagged
with its source filename. It performs no validation or cleaning -- that is
the responsibility of :mod:`vernacbridge.validator` and
:mod:`vernacbridge.cleaner` respectively. Keeping these concerns separate is
what lets new datasets be dropped into the ``datasets`` folder with zero
code changes: the loader will pick up any ``*.csv`` file automatically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetReadError,
    EmptyDatasetError,
    NoDatasetsFoundError,
)
from .utils import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Discovers and loads raw CSV datasets from a directory."""

    def __init__(self, dataset_dir: str | Path = config.DEFAULT_DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)

    def discover_csv_files(self) -> list[Path]:
        """
        Return every ``*.csv`` file directly inside ``dataset_dir``.

        The generated ``master_dataset.csv`` is excluded so that rebuilding
        the master dataset never accidentally folds a previous build back
        into itself.
        """
        if not self.dataset_dir.exists():
            raise DatasetDirectoryNotFoundError(
                f"Dataset directory does not exist: {self.dataset_dir}"
            )
        if not self.dataset_dir.is_dir():
            raise DatasetDirectoryNotFoundError(
                f"Dataset path is not a directory: {self.dataset_dir}"
            )

        master_path = Path(config.DEFAULT_MASTER_DATASET_PATH).resolve()
        csv_files = sorted(
            p
            for p in self.dataset_dir.glob("*.csv")
            if p.resolve() != master_path
        )

        if not csv_files:
            raise NoDatasetsFoundError(
                f"No CSV files found in dataset directory: {self.dataset_dir}"
            )

        logger.info("Discovered %d dataset file(s) in %s", len(csv_files), self.dataset_dir)
        return csv_files

    def load_csv(self, path: Path) -> pd.DataFrame:
        """
        Load a single CSV file, trying a small set of common encodings.

        Every row is tagged with a ``source_file`` column so validation and
        cleaning failures can be traced back to the originating dataset.
        """
        last_error: Exception | None = None

        for encoding in config.CANDIDATE_ENCODINGS:
            try:
                df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
                break
            except (UnicodeDecodeError, UnicodeError) as exc:
                last_error = exc
                continue
            except pd.errors.EmptyDataError as exc:
                raise EmptyDatasetError(f"Dataset file is empty: {path}") from exc
            except pd.errors.ParserError as exc:
                raise DatasetReadError(f"Could not parse CSV file {path}: {exc}") from exc
        else:
            raise DatasetReadError(
                f"Could not decode {path} with any of {config.CANDIDATE_ENCODINGS}"
            ) from last_error

        if df.empty:
            raise EmptyDatasetError(f"Dataset file has no data rows: {path}")

        df["source_file"] = path.name
        logger.info("Loaded %d row(s) from %s", len(df), path.name)
        return df

    def load_all(self) -> list[pd.DataFrame]:
        """Discover and load every dataset in ``dataset_dir``.

        Individual files that fail to load are logged and skipped rather
        than aborting the entire build, so one malformed CSV cannot block
        every other dataset from being merged.
        """
        frames: list[pd.DataFrame] = []
        for path in self.discover_csv_files():
            try:
                frames.append(self.load_csv(path))
            except (DatasetReadError, EmptyDatasetError) as exc:
                logger.error("Skipping %s: %s", path.name, exc)

        if not frames:
            raise NoDatasetsFoundError(
                f"No dataset file in {self.dataset_dir} could be loaded successfully."
            )

        return frames
