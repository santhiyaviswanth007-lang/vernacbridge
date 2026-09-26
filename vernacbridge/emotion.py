"""
Emotion Detection Engine.

Detects the emotion expressed in a piece of Tanglish text by finding the
closest matching sentence in the master dataset and returning its labelled
emotion.
"""

from __future__ import annotations

from typing import TypedDict

from . import config
from .core import MatchEngine
from .utils import get_logger

logger = get_logger(__name__)


class EmotionResult(TypedDict):
    emotion: str
    confidence: float
    matched_input: str


class EmotionDetector:
    """Detects the dominant emotion in Tanglish text via nearest-neighbor matching."""

    def __init__(self, engine: MatchEngine | None = None) -> None:
        self.engine = engine if engine is not None else MatchEngine()

    def detect_emotion(
        self, text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD
    ) -> EmotionResult:
        """Return the most likely emotion label for ``text``."""
        if not text or not text.strip():
            return EmotionResult(emotion="unknown", confidence=0.0, matched_input="")

        match = self.engine.best_match(text)

        if match is None or match.score < min_confidence:
            logger.info("No confident emotion match for: %r", text)
            return EmotionResult(
                emotion="unknown",
                confidence=0.0 if match is None else round(match.score, 2),
                matched_input="",
            )

        return EmotionResult(
            emotion=match.emotion,
            confidence=round(match.score, 2),
            matched_input=match.tanglish_input,
        )


_default_detector: EmotionDetector | None = None


def _get_default_detector() -> EmotionDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = EmotionDetector()
    return _default_detector


def detect_emotion(text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD) -> EmotionResult:
    """Module-level convenience function backing ``vernacbridge.detect_emotion()``."""
    return _get_default_detector().detect_emotion(text, min_confidence=min_confidence)
