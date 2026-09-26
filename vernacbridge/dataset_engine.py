"""
Dataset Engine.

This is the single entry point for turning the raw, per-domain CSV files in
``datasets/`` into one clean ``master_dataset.csv`` that the runtime engines
(:mod:`translator`, :mod:`emotion`, :mod:`intent`, :mod:`search`) consume.

Pipeline::

    DatasetLoader   -> discover + read every *.csv
    DatasetValidator -> per-file schema/label/empty-row validation
    pandas.concat    -> merge into one frame
    DatasetCleaner   -> exact + near-duplicate removal across the whole corpus
    -> master_dataset.csv + build_report.json

Adding a new domain dataset requires dropping a CSV into ``datasets/`` and
running this engine again -- no other code changes are needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .cleaner import CleaningReport, DatasetCleaner
from .exceptions import DatasetError, ValidationError
from .loader import DatasetLoader
from .utils import get_logger
from .validator import DatasetValidator, ValidationReport

logger = get_logger(__name__)


@dataclass
class BuildReport:
    """Full report of a single master dataset build."""

    files_loaded: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)
    total_rows_before_validation: int = 0
    total_rows_after_validation: int = 0
    total_rows_after_cleaning: int = 0
    per_file_validation: list[dict] = field(default_factory=list)
    cleaning: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class DatasetEngine:
    """Builds a single, clean ``master_dataset.csv`` from raw domain datasets."""

    def __init__(
        self,
        dataset_dir: str | Path = config.DEFAULT_DATASET_DIR,
        master_dataset_path: str | Path = config.DEFAULT_MASTER_DATASET_PATH,
        build_report_path: str | Path = config.DEFAULT_BUILD_REPORT_PATH,
        near_duplicate_threshold: float = config.NEAR_DUPLICATE_THRESHOLD,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.master_dataset_path = Path(master_dataset_path)
        self.build_report_path = Path(build_report_path)

        self.loader = DatasetLoader(dataset_dir=self.dataset_dir)
        self.validator = DatasetValidator()
        self.cleaner = DatasetCleaner(near_duplicate_threshold=near_duplicate_threshold)

    def build(self, write_output: bool = True) -> tuple[pd.DataFrame, BuildReport]:
        """
        Run the full pipeline and return the master DataFrame plus a report.

        Parameters
        ----------
        write_output:
            When ``True`` (default), writes ``master_dataset.csv`` and
            ``build_report.json`` to disk. Set to ``False`` to only compute
            the result in memory (useful for tests or dry runs).
        """
        report = BuildReport()
        validated_frames: list[pd.DataFrame] = []

        raw_frames = self.loader.load_all()

        for raw_df in raw_frames:
            source_file = raw_df["source_file"].iloc[0]
            try:
                clean_df, val_report = self.validator.validate(raw_df, source_file)
            except (DatasetError, ValidationError) as exc:
                logger.error("Validation failed for %s: %s", source_file, exc)
                report.files_failed.append(source_file)
                continue

            report.files_loaded.append(source_file)
            report.total_rows_before_validation += val_report.rows_in
            report.total_rows_after_validation += val_report.rows_out
            report.per_file_validation.append(val_report.as_dict())

            if not clean_df.empty:
                validated_frames.append(clean_df)

        if not validated_frames:
            raise DatasetError(
                "No dataset produced any valid rows after validation; "
                "master dataset was not built."
            )

        merged = pd.concat(validated_frames, ignore_index=True)
        cleaned, cleaning_report = self.cleaner.clean(merged)

        report.total_rows_after_cleaning = len(cleaned)
        report.cleaning = cleaning_report.as_dict()

        if write_output:
            self._write_outputs(cleaned, report)

        logger.info(
            "Dataset Engine build complete: %d file(s) merged, %d row(s) in master dataset",
            len(report.files_loaded),
            len(cleaned),
        )
        return cleaned, report

    def _write_outputs(self, df: pd.DataFrame, report: BuildReport) -> None:
        self.master_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.master_dataset_path, index=False, encoding="utf-8")
        logger.info("Wrote master dataset to %s", self.master_dataset_path)

        self.build_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.build_report_path, "w", encoding="utf-8") as fh:
            json.dump(report.as_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Wrote build report to %s", self.build_report_path)


def main() -> None:
    """CLI entry point: ``python -m vernacbridge.dataset_engine``."""
    engine = DatasetEngine()
    _, report = engine.build()
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
