"""
Dataset validation.

Validation is split into two tiers:

1. **Structural validation** (:meth:`DatasetValidator.validate_columns`) is
   fatal -- if a dataset is missing a required column there is no sensible
   way to use any row from it, so the whole file is rejected.
2. **Content validation** (emotions, intents, empty fields) is row-level --
   a single row with a typo'd emotion label doesn't invalidate the other
   999 rows in the same file, so offending rows are dropped and reported
   instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config
from .exceptions import MissingColumnError
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Summary of what a validation pass did to a single dataset."""

    source_file: str
    rows_in: int
    rows_out: int
    dropped_empty: int = 0
    dropped_invalid_emotion: int = 0
    dropped_invalid_intent: int = 0
    invalid_emotion_labels: set[str] = field(default_factory=set)
    invalid_intent_labels: set[str] = field(default_factory=set)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "dropped_empty": self.dropped_empty,
            "dropped_invalid_emotion": self.dropped_invalid_emotion,
            "dropped_invalid_intent": self.dropped_invalid_intent,
            "invalid_emotion_labels": sorted(self.invalid_emotion_labels),
            "invalid_intent_labels": sorted(self.invalid_intent_labels),
        }


class DatasetValidator:
    """Validates a single raw dataset DataFrame against the schema in ``config``."""

    def __init__(
        self,
        required_columns: tuple[str, ...] = config.REQUIRED_COLUMNS,
        non_nullable_columns: tuple[str, ...] = config.NON_NULLABLE_COLUMNS,
        valid_emotions: frozenset[str] = config.VALID_EMOTIONS,
        valid_intents: frozenset[str] = config.VALID_INTENTS,
    ) -> None:
        self.required_columns = required_columns
        self.non_nullable_columns = non_nullable_columns
        self.valid_emotions = valid_emotions
        self.valid_intents = valid_intents

    def validate_columns(self, df: pd.DataFrame, source_file: str) -> None:
        """Raise :class:`MissingColumnError` if any required column is absent."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise MissingColumnError(
                f"{source_file} is missing required column(s): {missing}"
            )

    def _mask_empty_rows(self, df: pd.DataFrame) -> pd.Series:
        empty = pd.Series(False, index=df.index)
        for col in self.non_nullable_columns:
            empty |= df[col].astype(str).str.strip().eq("")
        return empty

    def validate(self, df: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Run the full validation pipeline on a single dataset.

        Returns the cleaned DataFrame (invalid rows dropped) along with a
        :class:`ValidationReport` describing what was removed and why.
        """
        self.validate_columns(df, source_file)
        rows_in = len(df)

        report = ValidationReport(source_file=source_file, rows_in=rows_in, rows_out=rows_in)

        empty_mask = self._mask_empty_rows(df)
        report.dropped_empty = int(empty_mask.sum())
        df = df.loc[~empty_mask].copy()

        emotion_norm = df["emotion"].astype(str).str.strip().str.lower()
        invalid_emotion_mask = ~emotion_norm.isin(self.valid_emotions)
        report.dropped_invalid_emotion = int(invalid_emotion_mask.sum())
        report.invalid_emotion_labels = set(emotion_norm[invalid_emotion_mask].unique())
        df = df.loc[~invalid_emotion_mask].copy()
        emotion_norm = emotion_norm.loc[df.index]

        intent_norm = df["intent"].astype(str).str.strip().str.lower()
        invalid_intent_mask = ~intent_norm.isin(self.valid_intents)
        report.dropped_invalid_intent = int(invalid_intent_mask.sum())
        report.invalid_intent_labels = set(intent_norm[invalid_intent_mask].unique())
        df = df.loc[~invalid_intent_mask].copy()

        # Persist normalized labels back onto the frame.
        df["emotion"] = emotion_norm.loc[df.index]
        df["intent"] = intent_norm.loc[df.index]

        report.rows_out = len(df)

        if report.rows_dropped:
            logger.warning(
                "%s: dropped %d/%d row(s) during validation (%s)",
                source_file,
                report.rows_dropped,
                rows_in,
                report.as_dict(),
            )
        else:
            logger.info("%s: all %d row(s) passed validation", source_file, rows_in)

        return df, report
