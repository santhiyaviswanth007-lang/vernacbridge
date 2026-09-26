"""
Translation Pipeline (Phase 1) -- the single orchestration entry point.

Produces a single, richly-structured "understanding" of a Tanglish sentence
in one pass: normalization, corpus search, best-match selection, intent,
emotion, confidence, and English translation -- everything a downstream
consumer (e.g. a future LOVE AI integration) needs, without composing
Translator + EmotionDetector + IntentDetector separately (which would
re-run the same fuzzy match three times for the same input).

This module is additive: it does not change Translator's existing
translate() contract, which remains available and unchanged for callers
that only need a flat translation result.

Everything here is built on vernacbridge.core.MatchEngine -- the same
shared matching layer every other engine in this library already uses.

Extensibility
-------------
TranslationPipeline is the single entry point every future language-
processing module is meant to plug into: Context Engine (Phase 4), Memory
Engine (Phase 3), Plugin Manager (Phase 7), Conversation Engine (Phase 4 /
12). Each is accepted as an optional, keyword-only, dependency-injected
attribute -- ``None`` by default -- and, when present, is actually called
at a defined point in ``run()``:

    resolve context (ContextEngine.resolve)
            |
            v
    plugin pre-hook (PluginManager.before_match)
            |
            v
    normalize -> MatchEngine.top_matches            <- unchanged core
            |
            v
    build result (confident / low-confidence)
            |
            v
    plugin post-hook (PluginManager.after_match)
            |
            v
    memory.record (MemoryEngine)                     <- side effect only
            |
            v
    conversation.next_turn (ConversationEngine)       <- notification only

None of these four components exist yet -- with all four left as ``None``
(today's reality), every one of these steps is skipped and behavior is
identical to a pipeline with no dependencies at all. The contracts they
must satisfy live in :mod:`vernacbridge.protocols` as ``typing.Protocol``
classes, deliberately kept separate from this module so a real
implementation never needs to import or inherit from anything here.

A fifth collaborator, ``dataset_intelligence`` (Phase 2), is different: it
IS implemented, in :mod:`vernacbridge.intelligence`, and is default-
constructed automatically (unlike the four above). It only ever runs when
a caller explicitly passes ``explain=True`` to ``run()`` -- the default
(``explain=False``) path never calls it and never adds anything to the
returned dict, so existing callers see zero output or performance change.
See "Explain mode" below.

Error-handling is intentionally asymmetric:

- ``ContextEngine.resolve`` and ``PluginManager.before_match``/
  ``after_match`` can change *what gets matched or returned*. If one of
  these raises, the exception propagates -- silently continuing with an
  unresolved query could produce a confident-looking but wrong answer with
  no trace of the failure.
- ``MemoryEngine.record`` and ``ConversationEngine.next_turn`` are side
  effects layered on top of an already-valid result. Failures there are
  logged and swallowed: a broken memory store or notification failure must
  never cost the caller a valid translation they already have.

``normalizer`` is also injectable (defaulting to ``utils.normalize_text``),
since text normalization is itself a swappable dependency -- e.g. a future
multilingual expansion may need a different normalizer per language.

Explain mode
------------
Normal usage (``vb.translate_pipeline(text)`` or ``pipeline.run(text)``)
returns exactly the same lightweight ``PipelineResult`` shape Phase 1
always returned -- translation, intent, emotion, confidence, and so on.
Nothing about that default output changed in Phase 2.

Passing ``explain=True`` adds one extra key, ``"explanation"``, containing
a :class:`vernacbridge.intelligence.MatchExplanation`: the runner-up
candidate, the score margin between it and the winner, whether that margin
counts as a "close call", and the full ranked candidate list. This is
meant for developers debugging or auditing match quality, not for the
default translation consumer -- which is why it's opt-in rather than
always present.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, NotRequired, TypedDict

from . import config
from .core import Match, MatchEngine
from .intelligence import DatasetIntelligence, MatchExplanation
from .protocols import (
    ConversationEngineProtocol,
    ContextEngineProtocol,
    DatasetIntelligenceProtocol,
    MemoryEngineProtocol,
    PluginManagerProtocol,
)
from .utils import get_logger, normalize_text

logger = get_logger(__name__)

#: Signature every normalizer (default or injected) must satisfy.
NormalizerFn = Callable[[str], str]

_DATASET_SOURCE_RE = re.compile(r"^(?:vernacbridge_)?(?P<domain>.+?)(?:_\d+)?$")


def _domain_from_source(source_file: str) -> str:
    """
    Derive a clean dataset/domain label from a raw source filename.

    e.g. "vernacbridge_career_1000.csv" -> "career"

    Falls back to the filename stem if it doesn't match the conventional
    "vernacbridge_<domain>_<count>.csv" pattern, so unexpected filenames
    (e.g. from Phase 7's plugin-style dataset drops) never raise -- they
    just produce a less-tidy label instead of failing the pipeline.
    """
    if not source_file:
        return ""
    stem = Path(source_file).stem
    m = _DATASET_SOURCE_RE.match(stem)
    return m.group("domain") if m else stem


def _keywords_to_list(keywords: str) -> list[str]:
    """Split the comma-separated ``keywords`` column into a clean list."""
    if not keywords:
        return []
    return [k.strip() for k in keywords.split(",") if k.strip()]


def _match_to_candidate(match: Match) -> "MatchCandidate":
    return MatchCandidate(
        tanglish_input=match.tanglish_input,
        english_output=match.english_output,
        intent=match.intent,
        emotion=match.emotion,
        keywords=_keywords_to_list(match.keywords),
        urgency=match.urgency,
        confidence=round(match.score, 2),
        dataset_source=_domain_from_source(match.source_file),
    )


class MatchCandidate(TypedDict):
    """One scored candidate, used both as the confident result's basis and
    inside ``top_matches`` when confidence is too low to commit to one."""

    tanglish_input: str
    english_output: str
    intent: str
    emotion: str
    keywords: list[str]
    urgency: str
    confidence: float
    dataset_source: str


class PipelineResult(TypedDict):
    """
    Structured output of :meth:`TranslationPipeline.run`.

    ``explanation`` is intentionally declared with ``NotRequired``: it is
    only present in the dict when ``explain=True`` was passed to ``run()``.
    With the default ``explain=False``, the returned dict has exactly the
    same keys Phase 1 always returned -- no schema drift for callers who
    never opt in.
    """

    input: str
    normalized: str
    english_output: str
    intent: str
    emotion: str
    keywords: list[str]
    urgency: str
    confidence: float
    matched_sentence: str
    dataset_source: str
    is_confident: bool
    top_matches: list[MatchCandidate]
    session_id: str | None
    explanation: NotRequired[MatchExplanation | None]


def _empty_result(raw_text: str, normalized: str, session_id: str | None) -> PipelineResult:
    return PipelineResult(
        input=raw_text,
        normalized=normalized,
        english_output="",
        intent="unknown",
        emotion="unknown",
        keywords=[],
        urgency="unknown",
        confidence=0.0,
        matched_sentence="",
        dataset_source="",
        is_confident=False,
        top_matches=[],
        session_id=session_id,
    )


class TranslationPipeline:
    """
    Runs the full Phase 1 pipeline, with defined (but currently inert)
    extension points for future language-processing modules. See the
    module docstring for the full call order and error-handling rules.

    A single MatchEngine.top_matches() call backs the whole pipeline: the
    best candidate becomes the confident result, and the same call's
    remaining candidates back the low-confidence fallback, so no matching
    work is repeated between the two paths.

    Dependencies
    ------------
    engine, normalizer:
        Wired in and used today. ``engine`` defaults to a new
        :class:`MatchEngine`; ``normalizer`` defaults to
        :func:`vernacbridge.utils.normalize_text`.

    context_engine, memory_engine, plugin_manager, conversation_engine:
        Optional, keyword-only. Any object satisfying the corresponding
        Protocol in :mod:`vernacbridge.protocols` may be passed -- no
        inheritance required. All default to ``None``, in which case the
        corresponding step in ``run()`` is skipped entirely and behavior
        is unchanged from a pipeline with no dependencies at all.

    dataset_intelligence:
        Optional, keyword-only. Unlike the four above, this defaults to a
        real :class:`~vernacbridge.intelligence.DatasetIntelligence`
        instance (Phase 2 ships an implementation), not ``None`` -- so
        ``explain=True`` works without any extra wiring. Still injectable,
        so a future Plugin System can supply an alternate strategy.
    """

    def __init__(
        self,
        engine: MatchEngine | None = None,
        normalizer: NormalizerFn | None = None,
        *,
        context_engine: ContextEngineProtocol | None = None,
        memory_engine: MemoryEngineProtocol | None = None,
        plugin_manager: PluginManagerProtocol | None = None,
        conversation_engine: ConversationEngineProtocol | None = None,
        dataset_intelligence: DatasetIntelligenceProtocol | None = None,
    ) -> None:
        self.engine = engine if engine is not None else MatchEngine()
        self.normalizer: NormalizerFn = normalizer if normalizer is not None else normalize_text

        self.context_engine = context_engine
        self.memory_engine = memory_engine
        self.plugin_manager = plugin_manager
        self.conversation_engine = conversation_engine
        self.dataset_intelligence: DatasetIntelligenceProtocol = (
            dataset_intelligence if dataset_intelligence is not None else DatasetIntelligence()
        )

    def run(
        self,
        text: str,
        min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD,
        top_n: int = config.PIPELINE_TOP_N,
        session_id: str | None = None,
        explain: bool = False,
    ) -> PipelineResult:
        """
        Translate ``text`` and return a full structured result.

        If the best match's score is below ``min_confidence``, no single
        translation is committed to; instead ``is_confident`` is False and
        ``top_matches`` holds up to ``top_n`` closest candidates so the
        caller can decide how to proceed (e.g. show the user options).

        ``session_id`` identifies the conversation this call belongs to,
        for the benefit of injected context/memory/conversation
        components. It has no effect when none are injected.

        ``explain`` (default ``False``) is opt-in developer-facing detail:
        when ``True``, the returned dict gains an ``"explanation"`` key
        (see module docstring, "Explain mode"). When ``False``, the
        returned dict is byte-for-byte the same shape Phase 1 always
        returned, and DatasetIntelligence is never called.
        """
        raw_text = text or ""

        # --- Context Engine: resolve follow-ups against prior turns -----
        if self.context_engine is not None:
            raw_text = self.context_engine.resolve(raw_text, session_id) or raw_text

        # --- Plugin Manager: pre-match hook ------------------------------
        if self.plugin_manager is not None:
            raw_text = self.plugin_manager.before_match(raw_text) or raw_text

        normalized = self.normalizer(raw_text)

        if not normalized:
            logger.info("Empty input passed to TranslationPipeline")
            result = _empty_result(raw_text, normalized, session_id)
            if explain:
                result["explanation"] = None
            return self._finish(result, session_id)

        candidates = self.engine.top_matches(raw_text, n=max(top_n, 1))

        if not candidates:
            logger.info("Corpus produced no candidates for: %r", raw_text)
            result = _empty_result(raw_text, normalized, session_id)
            if explain:
                result["explanation"] = None
            return self._finish(result, session_id)

        best = candidates[0]

        if best.score < min_confidence:
            logger.info(
                "Low-confidence match for %r (best score %.2f < %.2f); "
                "returning top %d candidates instead of committing to one",
                raw_text,
                best.score,
                min_confidence,
                top_n,
            )
            result = PipelineResult(
                input=raw_text,
                normalized=normalized,
                english_output="",
                intent="unknown",
                emotion="unknown",
                keywords=[],
                urgency="unknown",
                confidence=round(best.score, 2),
                matched_sentence="",
                dataset_source="",
                is_confident=False,
                top_matches=[_match_to_candidate(m) for m in candidates[:top_n]],
                session_id=session_id,
            )
        else:
            result = PipelineResult(
                input=raw_text,
                normalized=normalized,
                english_output=best.english_output,
                intent=best.intent,
                emotion=best.emotion,
                keywords=_keywords_to_list(best.keywords),
                urgency=best.urgency,
                confidence=round(best.score, 2),
                matched_sentence=best.tanglish_input,
                dataset_source=_domain_from_source(best.source_file),
                is_confident=True,
                top_matches=[],
                session_id=session_id,
            )

        # --- Dataset Intelligence: explain the decision (opt-in only) ---
        # Skipped entirely when explain=False -- no call, no key added, no
        # cost. This is the only place DatasetIntelligence is ever invoked.
        if explain:
            result["explanation"] = self.dataset_intelligence.explain(candidates, top_n)

        return self._finish(result, session_id)

    def _finish(self, result: PipelineResult, session_id: str | None) -> PipelineResult:
        """
        Run the post-match extension points (plugin post-hook, memory,
        conversation) shared by every return path in ``run()``.
        """
        # --- Plugin Manager: post-match hook -----------------------------
        if self.plugin_manager is not None:
            result = self.plugin_manager.after_match(result) or result

        # --- Memory Engine: record the turn (side effect, non-fatal) ----
        if self.memory_engine is not None:
            try:
                self.memory_engine.record(session_id, result)
            except Exception:
                logger.exception(
                    "MemoryEngine.record failed for session_id=%r; "
                    "continuing without recording this turn",
                    session_id,
                )

        # --- Conversation Engine: notify (side effect, non-fatal) -------
        if self.conversation_engine is not None:
            try:
                self.conversation_engine.next_turn(session_id, result)
            except Exception:
                logger.exception(
                    "ConversationEngine.next_turn failed for session_id=%r; "
                    "continuing without notifying it of this turn",
                    session_id,
                )

        return result


# --------------------------------------------------------------------------
# Module-level convenience wrapper (used by vernacbridge.translate_pipeline())
# --------------------------------------------------------------------------
_default_pipeline: TranslationPipeline | None = None


def _get_default_pipeline() -> TranslationPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = TranslationPipeline()
    return _default_pipeline


def translate_pipeline(
    text: str,
    min_confidence: float = config.MATCH_CONFIDENCE_THRESHOLD,
    top_n: int = config.PIPELINE_TOP_N,
    session_id: str | None = None,
    explain: bool = False,
) -> PipelineResult:
    """Module-level convenience function backing ``vernacbridge.translate_pipeline()``."""
    return _get_default_pipeline().run(
        text, min_confidence=min_confidence, top_n=top_n, session_id=session_id, explain=explain
    )
