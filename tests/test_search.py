"""Tests for vernacbridge.search.SearchEngine."""

from __future__ import annotations

from vernacbridge.core import MatchEngine
from vernacbridge.search import SearchEngine


class TestSearchByEmotion:
    def test_finds_rows_with_matching_emotion(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        results = search.search_by_emotion("fear")
        assert len(results) == 1
        assert results[0]["emotion"] == "fear"

    def test_is_case_insensitive(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        results = search.search_by_emotion("FEAR")
        assert len(results) == 1

    def test_no_match_returns_empty_list(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        assert search.search_by_emotion("lonely") == []

    def test_respects_limit(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        results = search.search_by_emotion("fear", limit=0)
        assert results == []


class TestSearchByIntent:
    def test_finds_rows_with_matching_intent(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        results = search.search_by_intent("career")
        assert len(results) == 1
        assert results[0]["intent"] == "career"

    def test_no_match_returns_empty_list(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        assert search.search_by_intent("emergency") == []


class TestSearchByKeyword:
    def test_matches_keywords_column(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        results = search.search_by_keyword("interview")
        assert len(results) == 1
        assert "interview" in results[0]["keywords"]

    def test_matches_tanglish_input(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        results = search.search_by_keyword("santhosham")
        assert len(results) == 1

    def test_matches_english_output(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        results = search.search_by_keyword("afraid")
        assert len(results) == 1

    def test_empty_keyword_returns_empty_list(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        assert search.search_by_keyword("") == []

    def test_no_match_returns_empty_list(self, master_csv):
        search = SearchEngine(engine=MatchEngine(master_dataset_path=master_csv))
        assert search.search_by_keyword("zzz_nonexistent") == []
