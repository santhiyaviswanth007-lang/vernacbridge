"""
Dataset cleaning: exact and near-duplicate removal.

Near-duplicate removal uses RapidFuzz similarity scoring, but naive pairwise
comparison is O(n^2), which does not scale as the corpus grows into the
tens of thousands of rows the project vision calls for. To keep this
practical, rows are first bucketed by a cheap normalized prefix
(:func:`vernacbridge.utils.block_key`); fuzzy comparison only ever happens
*within* a bucket, since two sentences with completely different openings
are virtually never near-duplicates in this domain.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from . import config
from .utils import block_key, get_logger, normalize_text, similarity_score

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """Summary of what a cleaning pass removed from the merged dataset."""

    rows_in: int
    rows_out: int
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0

    @property
    def total_removed(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "total_removed": self.total_removed,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
        }


class DatasetCleaner:
    """Removes empty, exactly duplicated, and near-duplicate rows from a dataset."""

    def __init__(
        self,
        near_duplicate_threshold: float = config.NEAR_DUPLICATE_THRESHOLD,
        match_column: str = "tanglish_input",
    ) -> None:
        self.near_duplicate_threshold = near_duplicate_threshold
        self.match_column = match_column

    def remove_exact_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop rows that are byte-for-byte duplicates of an earlier row."""
        normalized = df[self.match_column].astype(str).map(normalize_text)
        dup_mask = normalized.duplicated(keep="first")
        removed = int(dup_mask.sum())
        return df.loc[~dup_mask].copy(), removed

    def remove_near_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Drop rows whose ``match_column`` value is a near-duplicate (RapidFuzz
        similarity >= threshold) of an earlier-seen row in the same bucket.
        The first occurrence of each near-duplicate cluster is kept.
        """
        buckets: dict[str, list[str]] = defaultdict(list)
        keep_mask: list[bool] = []

        for value in df[self.match_column].astype(str):
            normalized = normalize_text(value)
            key = block_key(value)
            bucket = buckets[key]

            is_near_duplicate = any(
                similarity_score(normalized, seen) >= self.near_duplicate_threshold
                for seen in bucket
            )

            keep_mask.append(not is_near_duplicate)
            if not is_near_duplicate:
                bucket.append(normalized)

        removed = keep_mask.count(False)
        return df.loc[keep_mask].copy(), removed

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """Run the full cleaning pipeline: exact dedup, then near-dedup."""
        rows_in = len(df)

        df, exact_removed = self.remove_exact_duplicates(df)
        df, near_removed = self.remove_near_duplicates(df)

        report = CleaningReport(
            rows_in=rows_in,
            rows_out=len(df),
            exact_duplicates_removed=exact_removed,
            near_duplicates_removed=near_removed,
        )
        logger.info("Cleaning complete: %s", report.as_dict())
        return df.reset_index(drop=True), report
