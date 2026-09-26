"""
Structural interfaces (``typing.Protocol``) that future VernacBridge
components must satisfy to be injected into :class:`~vernacbridge.pipeline.
TranslationPipeline` -- or any future orchestration layer -- without
requiring inheritance or tight coupling to this module.

Each protocol corresponds to a not-yet-implemented phase:

- ``ContextEngineProtocol``    -> Context Engine   (Phase 4)
- ``MemoryEngineProtocol``     -> Memory Engine     (Phase 3)
- ``PluginManagerProtocol``    -> Plugin Manager    (Phase 7)
- ``ConversationEngineProtocol`` -> Conversation Engine (Phase 4 / Phase 12)

One protocol, ``DatasetIntelligenceProtocol``, is different: it corresponds
to a component that IS implemented, in Phase 2 (:mod:`vernacbridge.
intelligence`). It's still expressed as a Protocol, and TranslationPipeline
still only depends on the interface, so a future Plugin System (Phase 7)
can supply an alternate explanation strategy without pipeline.py changing.

These are intentionally minimal and PROVISIONAL: they capture only what
TranslationPipeline currently needs from each collaborator so it can be
built and tested today. When each phase is actually designed, expect these
contracts to be refined -- most likely extended, ideally not broken -- not
treated as a finished spec for those modules.

Because these are ``Protocol`` classes, a future concrete implementation
(e.g. a real ``ContextEngine`` class) does not need to inherit from
anything here. It only needs matching method signatures. This keeps
``pipeline.py`` decoupled from implementations that don't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Avoids a runtime circular import: pipeline.py imports these protocols,
    # so protocols.py can only reference PipelineResult for type-checking.
    from .pipeline import PipelineResult
    from .intelligence import MatchExplanation
    from .core import Match


@runtime_checkable
class ContextEngineProtocol(Protocol):
    """
    Resolves conversational context -- pronouns, ellipsis, follow-up
    questions like "why?" -- into self-contained text the matching layer
    can act on.

    Implemented in Phase 4 by :class:`vernacbridge.context.ContextEngine`,
    which confirmed this signature (a plain string in, a plain string out)
    is sufficient for a mechanical, memory-backed resolution strategy.
    TranslationPipeline consumes only the resolved text.
    """

    def resolve(self, text: str, session_id: str | None = None) -> str:
        """
        Return ``text`` rewritten with relevant prior context folded in
        (e.g. "why?" -> "why am I afraid?"), or ``text`` unchanged if no
        relevant context exists or ``session_id`` is unknown.
        """
        ...


@runtime_checkable
class MemoryEngineProtocol(Protocol):
    """
    Records and recalls pipeline results for a conversation session.

    Provisional: Phase 3 explicitly calls for short-term, long-term, and
    session memory as distinct concerns; this single protocol may end up
    backed by a component that composes three separate stores, or may
    split into three protocols. TranslationPipeline only needs "record
    this result" and "recall history" today.
    """

    def record(self, session_id: str | None, result: "PipelineResult") -> None:
        """Persist ``result`` as part of ``session_id``'s history."""
        ...

    def recall(self, session_id: str | None) -> list["PipelineResult"]:
        """Return prior results recorded for ``session_id``, oldest first."""
        ...


@runtime_checkable
class PluginManagerProtocol(Protocol):
    """
    Hooks for extending the pipeline without modifying it (Phase 7).

    A plugin manager can rewrite the input before matching (e.g. dialect or
    spelling normalization contributed by a plugin) and/or enrich the
    result after matching (e.g. attaching data sourced from a newly
    dropped-in dataset), without TranslationPipeline knowing anything about
    individual plugins.
    """

    def before_match(self, text: str) -> str:
        """Return ``text``, optionally rewritten, before it reaches MatchEngine."""
        ...

    def after_match(self, result: "PipelineResult") -> "PipelineResult":
        """Return ``result``, optionally enriched or modified, before it's returned."""
        ...


@runtime_checkable
class ConversationEngineProtocol(Protocol):
    """
    Advances multi-turn conversation state (Phase 4, and ultimately
    Phase 12's LOVE AI integration).

    Provisional: a real implementation will likely need access to memory
    and context to do anything useful; this minimal hook exists so
    TranslationPipeline can notify it that a turn completed without
    depending on how it uses that notification.
    """

    def next_turn(self, session_id: str | None, result: "PipelineResult") -> None:
        """Notify the conversation engine that a turn completed."""
        ...


@runtime_checkable
class DatasetIntelligenceProtocol(Protocol):
    """
    Explains a match decision after it's been made (Phase 2). Given the
    scored candidates MatchEngine already produced for a query, returns a
    structured explanation of why the winner won -- the runner-up, the
    score margin between them, and whether the margin is narrow enough to
    be a "close call".

    Unlike the other protocols in this module, a real implementation of
    this one ships in this same phase (:class:`vernacbridge.intelligence.
    DatasetIntelligence`) -- this Protocol exists so TranslationPipeline
    depends on the interface rather than the concrete class, keeping the
    door open for a future Plugin System (Phase 7) to supply an alternate
    explanation strategy without pipeline.py changing again.

    Only called when a caller explicitly opts into ``explain=True`` --
    never on the default path, and therefore never a source of overhead
    for callers who don't ask for it.
    """

    def explain(
        self, candidates: list["Match"], top_n: int
    ) -> "MatchExplanation | None":
        """
        Return a structured explanation of ``candidates`` (already sorted
        best-first by MatchEngine), or ``None`` if ``candidates`` is empty.
        Performs no new matching -- pure analysis of data already in hand.
        """
        ...
