"""
Context Engine (Phase 4).

Implements :class:`vernacbridge.protocols.ContextEngineProtocol`, the
interface :class:`~vernacbridge.pipeline.TranslationPipeline` has called
into since Phase 1 (``self.context_engine.resolve(raw_text, session_id) or
raw_text``, before matching, with failures propagating) without any
concrete implementation existing until now.

Scope, deliberately limited
---------------------------
This is corpus-matching architecture, not a language model -- there is no
generative or semantic understanding underneath it. A "real" context
engine could mean deep coreference resolution or topic tracking; building
that here would be inventing linguistic capability for Tanglish specifically
that hasn't been validated, and is out of V1's scope regardless.

What ``ContextEngine.resolve()`` actually does is mechanical: if the
current input is short (a *structural* signal -- word count under
``config.CONTEXT_FOLLOWUP_MAX_WORDS`` -- not a guess at specific Tanglish
question-words, which would risk encoding vocabulary assumptions this
project hasn't validated) and there's a confident prior turn on record for
this session, the prior turn's matched Tanglish sentence is prepended to
the new text before it reaches MatchEngine. That's the entire mechanism:
give the existing fuzzy matcher more to work with, using only what was
already confidently matched before. No new capability is invented beyond
string composition over an existing memory read.

Design decision: ``memory_engine`` is required, not optional
--------------------------------------------------------------
Every other TranslationPipeline collaborator (``plugin_manager``,
``conversation_engine``, and ``memory_engine`` itself) defaults to
``None`` and is skipped when absent. ``ContextEngine`` does not follow
that pattern on purpose.

The first bug ever found in this project was ``self.engine = engine or
MatchEngine()`` -- falsy-engine-triggered silent creation of a
disconnected instance instead of sharing the intended one. ``ContextEngine``
is at direct risk of the same failure mode: its entire purpose is reading
memory that ``TranslationPipeline`` writes to elsewhere. If it were allowed
to default-construct its own private ``MemoryEngine()`` when none was
given, it would silently see empty history forever, and the failure would
look exactly like "context resolution just doesn't work" with no obvious
cause. Requiring ``memory_engine`` at construction forces the caller to be
explicit: the *same* ``MemoryEngine`` instance must be passed to both
``TranslationPipeline(memory_engine=m, context_engine=ContextEngine(memory_engine=m))``
(or the equivalent ``VernacBridge`` kwargs). See
``tests/test_context.py::TestSharedMemoryRequirement`` for a test that
specifically exercises the "two different instances" mistake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import config
from .protocols import MemoryEngineProtocol
from .utils import get_logger

if TYPE_CHECKING:
    from .pipeline import PipelineResult

logger = get_logger(__name__)


class ContextEngine:
    """
    Default implementation of :class:`vernacbridge.protocols.
    ContextEngineProtocol`.

    ``memory_engine`` is required -- see the module docstring for why this
    diverges from every other injectable collaborator in the pipeline.
    """

    def __init__(
        self,
        memory_engine: MemoryEngineProtocol,
        max_followup_words: int = config.CONTEXT_FOLLOWUP_MAX_WORDS,
        recency_window: int = config.CONTEXT_RECENCY_WINDOW,
    ) -> None:
        if memory_engine is None:
            raise ValueError(
                "ContextEngine requires a memory_engine -- pass the SAME "
                "MemoryEngine instance given to TranslationPipeline/"
                "VernacBridge, so context resolution reads the history "
                "that's actually being recorded. See the module docstring "
                "for why this argument has no default."
            )
        self.memory_engine = memory_engine
        self.max_followup_words = max_followup_words
        self.recency_window = recency_window

    def resolve(self, text: str, session_id: str | None = None) -> str:
        """
        Return ``text`` unchanged unless it looks like a short follow-up
        AND a usable (confident, previously-matched) prior turn exists for
        ``session_id`` -- in which case, that prior turn's matched
        Tanglish sentence is prepended.
        """
        stripped = (text or "").strip()

        if not stripped:
            return text

        if len(stripped.split()) > self.max_followup_words:
            return text  # looks like a complete sentence -- leave it alone

        if session_id is None:
            return text  # nothing to correlate history against

        history: list["PipelineResult"] = self.memory_engine.recall(session_id)
        if not history:
            return text

        recent = history[-self.recency_window:] if self.recency_window > 0 else []
        usable_sentences = [
            turn["matched_sentence"]
            for turn in recent
            if turn.get("is_confident") and turn.get("matched_sentence")
        ]

        if not usable_sentences:
            return text  # nothing confident enough to fuse in

        resolved = " ".join([*usable_sentences, stripped])
        logger.info(
            "Resolved follow-up for session_id=%r: %r -> %r",
            session_id,
            text,
            resolved,
        )
        return resolved
