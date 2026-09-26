"""
Dataset Intelligence (Phase 2).

Answers "why did the pipeline choose this match?" -- as a separate,
optional analysis layer over data TranslationPipeline already has, not as
a change to what the pipeline computes.

Design constraint driving everything here: normal library users only want
translation / intent / emotion / confidence, and Phase 1's output must stay
exactly that by default. DatasetIntelligence is a developer-facing tool,
activated only via ``TranslationPipeline.run(..., explain=True)``. When not
activated, this module isn't touched at all -- not called, not imported at
call time beyond the initial (cheap) object construction -- so there is no
performance cost for the default path.

Nothing here re-runs matching. DatasetIntelligence.explain() operates
purely on the list[Match] that MatchEngine.top_matches() already produced
for the current query -- it's analysis of an existing decision, not a new
one.
"""

from __future__ import annotations

from typing import TypedDict

from . import config
from .core import Match
from .utils import get_logger

logger = get_logger(__name__)


class CandidateExplanation(TypedDict):
    """One ranked candidate, as shown in a MatchExplanation."""

    rank: int
    tanglish_input: str
    english_output: str
    intent: str
    emotion: str
    dataset_source: str
    confidence: float


class MatchExplanation(TypedDict):
    """
    Structured explanation of why a particular candidate won the match.

    ``runner_up`` and ``margin`` are ``None`` when there was only one
    candidate to begin with -- nothing to compare the winner against.
    """

    chosen: CandidateExplanation
    runner_up: CandidateExplanation | None
    margin: float | None
    candidate_count: int
    is_close_call: bool
    ranking: list[CandidateExplanation]


def _to_candidate_explanation(match: Match, rank: int) -> CandidateExplanation:
    # Reuses the same domain-derivation convention as pipeline.py
    # (vernacbridge_career_1000.csv -> "career"), duplicated narrowly here
    # rather than imported to avoid a pipeline.py <-> intelligence.py
    # circular dependency; both derive from the same source_file field.
    from .pipeline import _domain_from_source

    return CandidateExplanation(
        rank=rank,
        tanglish_input=match.tanglish_input,
        english_output=match.english_output,
        intent=match.intent,
        emotion=match.emotion,
        dataset_source=_domain_from_source(match.source_file),
        confidence=round(match.score, 2),
    )


class DatasetIntelligence:
    """
    Default implementation of :class:`vernacbridge.protocols.
    DatasetIntelligenceProtocol`.

    Stateless and side-effect-free by default: ``explain()`` is a pure
    function of the candidates it's given. ``close_call_margin`` is
    injectable (defaults to ``config.CLOSE_CALL_MARGIN``) so a caller can
    tune what counts as "close" without needing a config.py edit.
    """

    def __init__(self, close_call_margin: float = config.CLOSE_CALL_MARGIN) -> None:
        self.close_call_margin = close_call_margin

    def explain(self, candidates: list[Match], top_n: int) -> MatchExplanation | None:
        """
        Build a :class:`MatchExplanation` from ``candidates`` (already
        sorted best-first by MatchEngine). Returns ``None`` if
        ``candidates`` is empty -- there is nothing to explain.
        """
        if not candidates:
            return None

        ranked = candidates[:top_n] if top_n > 0 else candidates
        ranking = [_to_candidate_explanation(m, i + 1) for i, m in enumerate(ranked)]

        chosen = ranking[0]
        runner_up = ranking[1] if len(ranking) > 1 else None
        margin = round(chosen["confidence"] - runner_up["confidence"], 2) if runner_up else None
        is_close_call = margin is not None and margin < self.close_call_margin

        explanation = MatchExplanation(
            chosen=chosen,
            runner_up=runner_up,
            margin=margin,
            candidate_count=len(candidates),
            is_close_call=is_close_call,
            ranking=ranking,
        )

        logger.info(
            "Explain: chosen=%r confidence=%.2f margin=%s close_call=%s candidates=%d",
            chosen["tanglish_input"],
            chosen["confidence"],
            margin,
            is_close_call,
            len(candidates),
        )
        return explanation
