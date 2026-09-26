"""
Search Engine.

Provides structured lookups over the master dataset -- by emotion, by
intent, or by free-text keyword -- as opposed to the single best-match
lookups used by the translation/emotion/intent engines.
"""

from __future__ import annotations

from . import config
from .core import MatchEngine
from .utils import get_logger, normalize_text

logger = get_logger(__name__)


class SearchEngine:
    """Structured search over the VernacBridge corpus."""

    def __init__(self, engine: MatchEngine | None = None) -> None:
        self.engine = engine if engine is not None else MatchEngine()

    def search_by_emotion(self, emotion: str, limit: int = 20) -> list[dict]:
        """Return up to ``limit`` rows labelled with the given emotion."""
        df = self.engine.dataframe
        target = emotion.strip().lower()
        matches = df.loc[df["emotion"].str.lower() == target]
        return matches.head(limit).to_dict(orient="records")

    def search_by_intent(self, intent: str, limit: int = 20) -> list[dict]:
        """Return up to ``limit`` rows labelled with the given intent."""
        df = self.engine.dataframe
        target = intent.strip().lower()
        matches = df.loc[df["intent"].str.lower() == target]
        return matches.head(limit).to_dict(orient="records")

    def search_by_keyword(self, keyword: str, limit: int = 20) -> list[dict]:
        """
        Return up to ``limit`` rows whose ``keywords`` field, Tanglish input,
        or English output contains ``keyword`` (case-insensitive substring
        match on normalized text).
        """
        df = self.engine.dataframe
        needle = normalize_text(keyword)
        if not needle:
            return []

        haystack = (
            df["keywords"].astype(str).map(normalize_text)
            + " "
            + df["tanglish_input"].astype(str).map(normalize_text)
            + " "
            + df["english_output"].astype(str).map(normalize_text)
        )
        matches = df.loc[haystack.str.contains(needle, regex=False)]
        return matches.head(limit).to_dict(orient="records")


_default_search_engine: SearchEngine | None = None


def _get_default_search_engine() -> SearchEngine:
    global _default_search_engine
    if _default_search_engine is None:
        _default_search_engine = SearchEngine()
    return _default_search_engine


def search_by_emotion(emotion: str, limit: int = 20) -> list[dict]:
    """Module-level convenience function backing ``vernacbridge.search_by_emotion()``."""
    return _get_default_search_engine().search_by_emotion(emotion, limit=limit)


def search_by_intent(intent: str, limit: int = 20) -> list[dict]:
    """Module-level convenience function backing ``vernacbridge.search_by_intent()``."""
    return _get_default_search_engine().search_by_intent(intent, limit=limit)


def search_by_keyword(keyword: str, limit: int = 20) -> list[dict]:
    """Module-level convenience function backing ``vernacbridge.search_by_keyword()``."""
    return _get_default_search_engine().search_by_keyword(keyword, limit=limit)
