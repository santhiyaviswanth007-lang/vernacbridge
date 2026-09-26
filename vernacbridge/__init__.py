"""
VernacBridge
============

A Python NLP library for understanding Tanglish (Tamil written in English
letters) -- translation, emotion detection, intent detection, and corpus
search, built on a continuously growing, human-labelled dataset.

Quick start
-----------
::

    import vernacbridge

    vernacbridge.translate("Enaku bayama iruku")
    vernacbridge.detect_emotion("Enaku bayama iruku")
    vernacbridge.detect_intent("Enaku bayama iruku")
    vernacbridge.search_by_emotion("fear")
    vernacbridge.translate_pipeline("Enaku bayama iruku")

Developers debugging match quality can opt into Phase 2's explanation data
without changing the default output for everyone else::

    vernacbridge.translate_pipeline("Enaku bayama iruku", explain=True)

To remember conversation turns across calls, inject a Phase 3
:class:`MemoryEngine` -- it stays opt-in, so nothing is recorded unless
you ask for it::

    from vernacbridge import MemoryEngine

    memory = MemoryEngine()
    vb = VernacBridge(memory_engine=memory)
    vb.translate_pipeline("Enaku bayama iruku", session_id="user-1")
    memory.recall("user-1")

For multi-instance or custom-dataset use cases, use the :class:`VernacBridge`
facade instead of the module-level functions::

    from vernacbridge import VernacBridge

    vb = VernacBridge(master_dataset_path="path/to/master_dataset.csv")
    vb.translate("Enaku bayama iruku")

To (re)build the master dataset from the raw per-domain CSVs::

    from vernacbridge import DatasetEngine

    DatasetEngine().build()
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .context import ContextEngine
from .core import Match, MatchEngine
from .dataset_engine import BuildReport, DatasetEngine
from .emotion import EmotionDetector, EmotionResult
from .emotion import detect_emotion as _module_detect_emotion
from .intent import IntentDetector, IntentResult
from .intent import detect_intent as _module_detect_intent
from .intelligence import DatasetIntelligence, MatchExplanation
from .memory import InMemoryStore, JSONFileStore, MemoryEngine, MemoryStore
from .pipeline import MatchCandidate, PipelineResult, TranslationPipeline
from .pipeline import translate_pipeline as _module_translate_pipeline
from .protocols import (
    ConversationEngineProtocol,
    ContextEngineProtocol,
    DatasetIntelligenceProtocol,
    MemoryEngineProtocol,
    PluginManagerProtocol,
)
from .search import SearchEngine
from .search import search_by_emotion as _module_search_by_emotion
from .search import search_by_intent as _module_search_by_intent
from .search import search_by_keyword as _module_search_by_keyword
from .translator import Translator, TranslationResult
from .translator import translate as _module_translate

__version__ = "1.0.0"

__all__ = [
    "VernacBridge",
    "DatasetEngine",
    "BuildReport",
    "MatchEngine",
    "Match",
    "Translator",
    "TranslationResult",
    "EmotionDetector",
    "EmotionResult",
    "IntentDetector",
    "IntentResult",
    "SearchEngine",
    "TranslationPipeline",
    "PipelineResult",
    "MatchCandidate",
    "DatasetIntelligence",
    "MatchExplanation",
    "MemoryEngine",
    "MemoryStore",
    "InMemoryStore",
    "JSONFileStore",
    "ContextEngine",
    "ContextEngineProtocol",
    "MemoryEngineProtocol",
    "PluginManagerProtocol",
    "ConversationEngineProtocol",
    "DatasetIntelligenceProtocol",
    "translate",
    "detect_emotion",
    "detect_intent",
    "search_by_emotion",
    "search_by_intent",
    "search_by_keyword",
    "translate_pipeline",
]


class VernacBridge:
    """
    Facade over every VernacBridge engine, sharing a single loaded corpus.

    Prefer this class over the module-level convenience functions whenever
    you need a non-default dataset path, multiple independent instances
    (e.g. in a multi-tenant service), or explicit control over when the
    corpus is loaded into memory.
    """

    def __init__(
        self,
        master_dataset_path: str | Path = config.DEFAULT_MASTER_DATASET_PATH,
        *,
        context_engine: ContextEngineProtocol | None = None,
        memory_engine: MemoryEngineProtocol | None = None,
        plugin_manager: PluginManagerProtocol | None = None,
        conversation_engine: ConversationEngineProtocol | None = None,
        dataset_intelligence: DatasetIntelligenceProtocol | None = None,
    ) -> None:
        self.engine = MatchEngine(master_dataset_path=master_dataset_path)
        self.translator = Translator(engine=self.engine)
        self.emotion_detector = EmotionDetector(engine=self.engine)
        self.intent_detector = IntentDetector(engine=self.engine)
        self.search_engine = SearchEngine(engine=self.engine)
        self.pipeline = TranslationPipeline(
            engine=self.engine,
            context_engine=context_engine,
            memory_engine=memory_engine,
            plugin_manager=plugin_manager,
            conversation_engine=conversation_engine,
            dataset_intelligence=dataset_intelligence,
        )

    def load(self) -> "VernacBridge":
        """Eagerly load the corpus (otherwise it lazy-loads on first use)."""
        self.engine.load()
        return self

    def translate(self, text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD) -> TranslationResult:
        return self.translator.translate(text, min_confidence=min_confidence)

    def detect_emotion(self, text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD) -> EmotionResult:
        return self.emotion_detector.detect_emotion(text, min_confidence=min_confidence)

    def detect_intent(self, text: str, min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD) -> IntentResult:
        return self.intent_detector.detect_intent(text, min_confidence=min_confidence)

    def search_by_emotion(self, emotion: str, limit: int = 20) -> list[dict]:
        return self.search_engine.search_by_emotion(emotion, limit=limit)

    def search_by_intent(self, intent: str, limit: int = 20) -> list[dict]:
        return self.search_engine.search_by_intent(intent, limit=limit)

    def search_by_keyword(self, keyword: str, limit: int = 20) -> list[dict]:
        return self.search_engine.search_by_keyword(keyword, limit=limit)

    def translate_pipeline(
        self,
        text: str,
        min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD,
        top_n: int = config.PIPELINE_TOP_N,
        session_id: str | None = None,
        explain: bool = False,
    ) -> PipelineResult:
        """
        Run the full Phase 1 structured pipeline: normalize, search, best-
        match selection, intent, emotion, confidence, translation -- in one
        call. Falls back to the top ``top_n`` candidates instead of a single
        answer when confidence is below ``min_confidence``.

        Pass ``explain=True`` for the Phase 2 developer-facing explanation
        (runner-up, score margin, full ranking) attached under the
        ``"explanation"`` key. Default output is unchanged either way.
        """
        return self.pipeline.run(
            text,
            min_confidence=min_confidence,
            top_n=top_n,
            session_id=session_id,
            explain=explain,
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions (backed by lazily-initialized singletons
# defined in each engine module). These are what `import vernacbridge;
# vernacbridge.translate(...)` calls.
# ---------------------------------------------------------------------------
translate = _module_translate
detect_emotion = _module_detect_emotion
detect_intent = _module_detect_intent
search_by_emotion = _module_search_by_emotion
search_by_intent = _module_search_by_intent
search_by_keyword = _module_search_by_keyword
translate_pipeline = _module_translate_pipeline
