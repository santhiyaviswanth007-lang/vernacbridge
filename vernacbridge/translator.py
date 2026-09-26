"""
Translation Engine.

Translates Tanglish (Tamil written in English letters) into English by
finding the closest known sentence in the master dataset via
:class:`vernacbridge.core.MatchEngine`.

This is a corpus-matching translator, not a generative one: it is exact and
explainable, which matters for a v1 built on a few thousand curated rows.
The architecture (see ``core.py``) is intentionally decoupled from *how*
matching happens, so a future statistical or LLM-based translator can be
substituted later without touching call sites like
``vernacbridge.translate(...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from . import config
from .core import MatchEngine
from .utils import get_logger

logger = get_logger(__name__)


class TranslationResult(TypedDict):
    translation: str
    confidence: float
    emotion: str
    intent: str
    matched_input: str
    urgency: str


class Translator:
    """Translates Tanglish text to English using nearest-neighbor matching."""

    def __init__(self, engine: MatchEngine | None = None) -> None:
        self.engine = engine if engine is not None else MatchEngine()

    def translate(
        self, text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD
    ) -> TranslationResult:
        """
        Translate ``text`` into English.

        If no known sentence is similar enough (score below
        ``min_confidence``), the original text is returned untranslated with
        the low confidence score intact, so callers can detect and handle
        the "out of vocabulary" case rather than silently receiving a wrong
        translation.
        """
        if not text or not text.strip():
            return TranslationResult(
                translation="",
                confidence=0.0,
                emotion="unknown",
                intent="unknown",
                matched_input="",
                urgency="unknown",
            )

        match = self.engine.best_match(text)

        if match is None or match.score < min_confidence:
            logger.info("No confident translation match for: %r", text)
            return TranslationResult(
                translation=text,
                confidence=0.0 if match is None else round(match.score, 2),
                emotion="unknown",
                intent="unknown",
                matched_input="",
                urgency="unknown",
            )

        return TranslationResult(
            translation=match.english_output,
            confidence=round(match.score, 2),
            emotion=match.emotion,
            intent=match.intent,
            matched_input=match.tanglish_input,
            urgency=match.urgency,
        )


# --------------------------------------------------------------------------
# Module-level convenience wrapper (used by vernacbridge.translate())
# --------------------------------------------------------------------------
_default_translator: Translator | None = None


def _get_default_translator() -> Translator:
    global _default_translator
    if _default_translator is None:
        _default_translator = Translator()
    return _default_translator


def translate(text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD) -> TranslationResult:
    """Module-level convenience function backing ``vernacbridge.translate()``."""
    return _get_default_translator().translate(text, min_confidence=min_confidence)
