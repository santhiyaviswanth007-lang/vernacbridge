"""Tests for vernacbridge.core.MatchEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from vernacbridge.core import Match, MatchEngine
from vernacbridge.exceptions import MasterDatasetNotBuiltError


class TestLoad:
    def test_raises_when_master_dataset_missing(self, tmp_path: Path):
        engine = MatchEngine(master_dataset_path=tmp_path / "nope.csv")
        with pytest.raises(MasterDatasetNotBuiltError):
            engine.load()

    def test_loads_row_count(self, master_csv, clean_rows):
        engine = MatchEngine(master_dataset_path=master_csv)
        engine.load()
        assert len(engine) == len(clean_rows)
        assert engine.is_loaded

    def test_lazy_loads_on_first_use(self, master_csv):
        engine = MatchEngine(master_dataset_path=master_csv)
        assert not engine.is_loaded
        engine.best_match("enaku bayama iruku")
        assert engine.is_loaded


class TestMatching:
    def test_exact_match_scores_100(self, master_csv):
        engine = MatchEngine(master_dataset_path=master_csv)
        match = engine.best_match("enaku bayama iruku")
        assert match is not None
        assert isinstance(match, Match)
        assert match.score == 100.0
        assert match.english_output == "I am afraid."

    def test_near_match_returns_correct_row(self, master_csv):
        engine = MatchEngine(master_dataset_path=master_csv)
        match = engine.best_match("enaku konjam bayama iruku")
        assert match is not None
        assert match.emotion == "fear"
        assert match.score > 60.0

    def test_top_matches_ordered_best_first(self, master_csv):
        engine = MatchEngine(master_dataset_path=master_csv)
        matches = engine.top_matches("enaku bayama iruku", n=3)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_top_matches_respects_n(self, master_csv):
        engine = MatchEngine(master_dataset_path=master_csv)
        matches = engine.top_matches("enaku bayama iruku", n=2)
        assert len(matches) == 2

    def test_match_as_dict_has_expected_keys(self, master_csv):
        engine = MatchEngine(master_dataset_path=master_csv)
        match = engine.best_match("enaku bayama iruku")
        d = match.as_dict()
        assert set(d) == {
            "tanglish_input",
            "english_output",
            "intent",
            "emotion",
            "keywords",
            "urgency",
            "confidence",
            "source_file",
            "score",
        }
