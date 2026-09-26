"""
Core matching engine.

All runtime-facing engines (:mod:`translator`, :mod:`emotion`, :mod:`intent`,
:mod:`search`) need the same thing: an in-memory, normalized view of the
master dataset plus a way to find the best (or top-N) matching row for a
piece of Tanglish text. That logic lives here, once, in :class:`MatchEngine`,
and is composed into the higher-level engines rather than duplicated across
them.

Keeping this as a separate layer also means future upgrades -- e.g. swapping
RapidFuzz matching for an embedding-based semantic search, or adding a
context/memory layer -- only need to change this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .exceptions import MasterDatasetNotBuiltError
from .utils import block_key, get_logger, normalize_text, similarity_score

logger = get_logger(__name__)


@dataclass(frozen=True)
class Match:
    """A single scored match from the corpus."""

    tanglish_input: str
    english_output: str
    intent: str
    emotion: str
    keywords: str
    urgency: str
    confidence: str
    source_file: str
    score: float

    def as_dict(self) -> dict:
        return {
            "tanglish_input": self.tanglish_input,
            "english_output": self.english_output,
            "intent": self.intent,
            "emotion": self.emotion,
            "keywords": self.keywords,
            "urgency": self.urgency,
            "confidence": self.confidence,
            "source_file": self.source_file,
            "score": round(self.score, 2),
        }


class MatchEngine:
    """
    Loads the master dataset and answers "what is the closest known
    sentence to this input?" queries.
    """

    def __init__(self, master_dataset_path: str | Path = config.DEFAULT_MASTER_DATASET_PATH) -> None:
        self.master_dataset_path = Path(master_dataset_path)
        self._df: pd.DataFrame | None = None
        self._normalized: list[str] = []
        self._buckets: dict[str, list[int]] = {}

    # ----------------------------------------------------------------
    # Loading
    # ----------------------------------------------------------------
    def load(self) -> None:
        """Load (or reload) the master dataset from disk into memory."""
        if not self.master_dataset_path.exists():
            raise MasterDatasetNotBuiltError(
                f"Master dataset not found at {self.master_dataset_path}. "
                "Run vernacbridge.dataset_engine.DatasetEngine().build() first."
            )

        df = pd.read_csv(self.master_dataset_path, dtype=str, keep_default_na=False)
        self._df = df.reset_index(drop=True)
        self._normalized = [normalize_text(t) for t in self._df["tanglish_input"]]

        buckets: dict[str, list[int]] = {}
        for idx, text in enumerate(self._df["tanglish_input"]):
            buckets.setdefault(block_key(text), []).append(idx)
        self._buckets = buckets

        logger.info("MatchEngine loaded %d row(s) from %s", len(df), self.master_dataset_path)

    @property
    def is_loaded(self) -> bool:
        return self._df is not None

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        assert self._df is not None
        return self._df

    def __len__(self) -> int:
        return 0 if self._df is None else len(self._df)

    # ----------------------------------------------------------------
    # Matching
    # ----------------------------------------------------------------
    def _row_to_match(self, idx: int, score: float) -> Match:
        row = self._df.iloc[idx]  # type: ignore[union-attr]
        return Match(
            tanglish_input=row["tanglish_input"],
            english_output=row["english_output"],
            intent=row["intent"],
            emotion=row["emotion"],
            keywords=row.get("keywords", ""),
            urgency=row.get("urgency", ""),
            confidence=row.get("confidence", ""),
            source_file=row.get("source_file", ""),
            score=score,
        )

    def _candidate_indices(self, text: str) -> list[int]:
        """
        Return indices to score against for a query, preferring the query's
        own bucket but falling back to the full corpus if the bucket is
        small -- short Tanglish phrases can legitimately start differently
        (e.g. spelling variants) while still meaning the same thing.
        """
        key = block_key(text)
        bucket = self._buckets.get(key, [])
        if len(bucket) >= 5:
            return bucket
        return list(range(len(self._df)))  # type: ignore[arg-type]

    def top_matches(self, text: str, n: int = 5) -> list[Match]:
        """Return the top ``n`` matches for ``text``, best first."""
        self._ensure_loaded()
        normalized_query = normalize_text(text)

        scored: list[tuple[float, int]] = []
        for idx in self._candidate_indices(text):
            score = similarity_score(normalized_query, self._normalized[idx])
            scored.append((score, idx))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._row_to_match(idx, score) for score, idx in scored[:n]]

    def best_match(self, text: str) -> Match | None:
        """Return the single best match for ``text``, or ``None`` if the corpus is empty."""
        matches = self.top_matches(text, n=1)
        return matches[0] if matches else None

    # ----------------------------------------------------------------
    # Raw access (used by search.py)
    # ----------------------------------------------------------------
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._ensure_loaded()
