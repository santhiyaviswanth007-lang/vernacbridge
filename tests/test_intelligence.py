"""Tests for vernacbridge.intelligence.DatasetIntelligence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vernacbridge.core import MatchEngine
from vernacbridge.intelligence import DatasetIntelligence


@pytest.fixture
def engine(master_csv: Path) -> MatchEngine:
    return MatchEngine(master_dataset_path=master_csv)


@pytest.fixture
def intelligence() -> DatasetIntelligence:
    return DatasetIntelligence()


class TestEmptyCandidates:
    def test_no_candidates_returns_none(self, intelligence):
        assert intelligence.explain([], top_n=3) is None


class TestSingleCandidate:
    def test_single_candidate_has_no_runner_up_or_margin(self, engine, intelligence):
        # threshold=0 guarantees at least the single closest match is
        # kept even if scored low, so we control candidate count directly
        # via top_n=1 regardless of corpus size.
        candidates = engine.top_matches("enaku bayama iruku", n=1)
        explanation = intelligence.explain(candidates, top_n=1)

        assert explanation["runner_up"] is None
        assert explanation["margin"] is None
        assert explanation["is_close_call"] is False
        assert explanation["candidate_count"] == 1
        assert len(explanation["ranking"]) == 1
        assert explanation["chosen"]["rank"] == 1


class TestMultipleCandidates:
    def test_chosen_and_runner_up_reflect_top_two_by_score(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)

        assert explanation["chosen"]["confidence"] == round(candidates[0].score, 2)
        assert explanation["runner_up"]["confidence"] == round(candidates[1].score, 2)
        assert explanation["candidate_count"] == len(candidates)

    def test_margin_is_chosen_minus_runner_up(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)

        expected_margin = round(
            explanation["chosen"]["confidence"] - explanation["runner_up"]["confidence"], 2
        )
        assert explanation["margin"] == expected_margin
        assert explanation["margin"] >= 0  # candidates are sorted best-first

    def test_ranking_is_sorted_best_first(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)

        scores = [c["confidence"] for c in explanation["ranking"]]
        assert scores == sorted(scores, reverse=True)

    def test_ranking_ranks_are_1_indexed_and_sequential(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)
        assert [c["rank"] for c in explanation["ranking"]] == list(
            range(1, len(explanation["ranking"]) + 1)
        )

    def test_top_n_limits_ranking_even_with_more_candidates_available(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=1)

        # candidate_count still reflects everything the caller passed in --
        # only the ranking/chosen/runner_up are limited by top_n.
        assert explanation["candidate_count"] == len(candidates)
        assert len(explanation["ranking"]) == 1
        assert explanation["runner_up"] is None  # only one candidate was ranked


class TestCloseCallThreshold:
    def test_margin_below_threshold_is_flagged_close_call(self, engine):
        intelligence = DatasetIntelligence(close_call_margin=1000.0)  # everything is "close"
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)
        assert explanation["is_close_call"] is True

    def test_margin_at_or_above_threshold_is_not_close_call(self, engine):
        intelligence = DatasetIntelligence(close_call_margin=-1.0)  # nothing is "close"
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)
        assert explanation["is_close_call"] is False

    def test_default_threshold_comes_from_config(self):
        from vernacbridge import config

        assert DatasetIntelligence().close_call_margin == config.CLOSE_CALL_MARGIN


class TestCandidateShape:
    def test_candidate_explanation_has_expected_keys(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)
        expected_keys = {
            "rank", "tanglish_input", "english_output", "intent",
            "emotion", "dataset_source", "confidence",
        }
        assert set(explanation["chosen"].keys()) == expected_keys

    def test_dataset_source_is_derived_from_source_file(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=1)
        explanation = intelligence.explain(candidates, top_n=1)
        # fixture rows use source_file="sample.csv" -> domain "sample"
        assert explanation["chosen"]["dataset_source"] == "sample"


class TestJSONSerializable:
    def test_full_explanation_is_json_serializable(self, engine, intelligence):
        candidates = engine.top_matches("enaku bayama iruku", n=3)
        explanation = intelligence.explain(candidates, top_n=3)
        # Must not raise -- this is meant to be logged/transmitted as-is.
        json.dumps(explanation)


class TestNoMatchingRecomputation:
    def test_explain_does_not_call_the_engine(self, engine, intelligence, monkeypatch):
        # explain() must be pure analysis of the candidates it's given --
        # never a fresh MatchEngine call of its own.
        candidates = engine.top_matches("enaku bayama iruku", n=3)

        def _forbidden(*args, **kwargs):
            raise AssertionError("DatasetIntelligence.explain() must not call MatchEngine")

        monkeypatch.setattr(engine, "top_matches", _forbidden)
        monkeypatch.setattr(engine, "best_match", _forbidden)

        intelligence.explain(candidates, top_n=3)  # must not raise
