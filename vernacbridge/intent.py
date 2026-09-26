"""
Intent Detection Engine.

Detects the conversational intent/domain (e.g. ``career``, ``family``,
``emergency``) of a piece of Tanglish text by finding the closest matching
sentence in the master dataset and returning its labelled intent.
"""

from __future__ import annotations

from typing import TypedDict

from . import config
from .core import MatchEngine
from .utils import get_logger

logger = get_logger(__name__)


class IntentResult(TypedDict):
    intent: str
    confidence: float
    matched_input: str
    urgency: str


class IntentDetector:
    """Detects the conversational intent of Tanglish text via nearest-neighbor matching."""

    def __init__(self, engine: MatchEngine | None = None) -> None:
        self.engine = engine if engine is not None else MatchEngine()

    def detect_intent(
        self, text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD
    ) -> IntentResult:
        """Return the most likely intent label for ``text``."""
        if not text or not text.strip():
            return IntentResult(intent="unknown", confidence=0.0, matched_input="", urgency="unknown")

        match = self.engine.best_match(text)

        if match is None or match.score < min_confidence:
            logger.info("No confident intent match for: %r", text)
            return IntentResult(
                intent="unknown",
                confidence=0.0 if match is None else round(match.score, 2),
                matched_input="",
                urgency="unknown",
            )

        return IntentResult(
            intent=match.intent,
            confidence=round(match.score, 2),
            matched_input=match.tanglish_input,
            urgency=match.urgency,
        )


_default_detector: IntentDetector | None = None


def _get_default_detector() -> IntentDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = IntentDetector()
    return _default_detector


def detect_intent(text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD) -> IntentResult:
    """Module-level convenience function backing ``vernacbridge.detect_intent()``."""
    return _get_default_detector().detect_intent(text, min_confidence=min_confidence)
