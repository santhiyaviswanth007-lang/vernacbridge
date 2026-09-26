"""
Tests for the corpus-matching runtime engines: Translator, EmotionDetector,
and IntentDetector. All three share the same shape (best-match lookup with
a confidence floor), so their test structure intentionally mirrors that.
"""

from __future__ import annotations

from vernacbridge.core import MatchEngine
from vernacbridge.emotion import EmotionDetector
from vernacbridge.intent import IntentDetector
from vernacbridge.translator import Translator


class TestTranslator:
    def test_confident_match_returns_translation(self, master_csv):
        translator = Translator(engine=MatchEngine(master_dataset_path=master_csv))
        result = translator.translate("enaku bayama iruku")
        assert result["translation"] == "I am afraid."
        assert result["emotion"] == "fear"
        assert result["intent"] == "mental_health"
        assert result["confidence"] == 100.0

    def test_low_confidence_returns_original_text_untranslated(self, master_csv):
        translator = Translator(engine=MatchEngine(master_dataset_path=master_csv))
        result = translator.translate("completely unrelated gibberish zzqx", min_confidence=99.0)
        assert result["translation"] == "completely unrelated gibberish zzqx"
        assert result["emotion"] == "unknown"
        assert result["intent"] == "unknown"

    def test_empty_input_returns_zero_confidence(self, master_csv):
        translator = Translator(engine=MatchEngine(master_dataset_path=master_csv))
        result = translator.translate("   ")
        assert result["translation"] == ""
        assert result["confidence"] == 0.0

    def test_min_confidence_threshold_is_respected(self, master_csv):
        translator = Translator(engine=MatchEngine(master_dataset_path=master_csv))
        # A near (not exact) match: should pass a lenient threshold...
        lenient = translator.translate("enaku konjam bayama iruku", min_confidence=50.0)
        assert lenient["translation"] == "I am afraid."
        # ...but fail an unreasonably strict one.
        strict = translator.translate("enaku konjam bayama iruku", min_confidence=99.9)
        assert strict["emotion"] == "unknown"


class TestEmotionDetector:
    def test_detects_known_emotion(self, master_csv):
        detector = EmotionDetector(engine=MatchEngine(master_dataset_path=master_csv))
        result = detector.detect_emotion("naan romba santhosham ah irukken")
        assert result["emotion"] == "happy"
        assert result["confidence"] == 100.0

    def test_empty_input_returns_unknown(self, master_csv):
        detector = EmotionDetector(engine=MatchEngine(master_dataset_path=master_csv))
        result = detector.detect_emotion("")
        assert result["emotion"] == "unknown"
        assert result["confidence"] == 0.0

    def test_low_confidence_input_returns_unknown(self, master_csv):
        detector = EmotionDetector(engine=MatchEngine(master_dataset_path=master_csv))
        result = detector.detect_emotion("xyz totally unrelated", min_confidence=99.0)
        assert result["emotion"] == "unknown"


class TestIntentDetector:
    def test_detects_known_intent(self, master_csv):
        detector = IntentDetector(engine=MatchEngine(master_dataset_path=master_csv))
        result = detector.detect_intent("interview ku poren, romba nervous ah irukku")
        assert result["intent"] == "career"
        assert result["urgency"] == "medium"

    def test_empty_input_returns_unknown(self, master_csv):
        detector = IntentDetector(engine=MatchEngine(master_dataset_path=master_csv))
        result = detector.detect_intent("")
        assert result["intent"] == "unknown"
        assert result["urgency"] == "unknown"

    def test_low_confidence_input_returns_unknown(self, master_csv):
        detector = IntentDetector(engine=MatchEngine(master_dataset_path=master_csv))
        result = detector.detect_intent("xyz totally unrelated", min_confidence=99.0)
        assert result["intent"] == "unknown"


class TestSharedEngineInstance:
    """Verifies engines can share one MatchEngine (as VernacBridge facade does) without conflict."""

    def test_translator_and_emotion_detector_share_engine_state(self, master_csv):
        shared = MatchEngine(master_dataset_path=master_csv)
        translator = Translator(engine=shared)
        detector = EmotionDetector(engine=shared)

        translator.translate("enaku bayama iruku")  # triggers lazy load
        assert shared.is_loaded

        result = detector.detect_emotion("enaku bayama iruku")
        assert result["emotion"] == "fear"
