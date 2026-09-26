"""Tests for vernacbridge.utils: normalization, bucketing, similarity scoring."""

from __future__ import annotations

from vernacbridge.utils import block_key, is_rapidfuzz_available, normalize_text, similarity_score


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Enaku Bayama") == "enaku bayama"

    def test_strips_punctuation(self):
        assert normalize_text("enaku, bayama!! iruku?") == "enaku bayama iruku"

    def test_collapses_whitespace(self):
        assert normalize_text("enaku    bayama\tiruku") == "enaku bayama iruku"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  enaku bayama  ") == "enaku bayama"

    def test_non_string_input_returns_empty_string(self):
        assert normalize_text(None) == ""  # type: ignore[arg-type]
        assert normalize_text(float("nan")) == ""  # type: ignore[arg-type]

    def test_empty_string_stays_empty(self):
        assert normalize_text("") == ""


class TestBlockKey:
    def test_returns_prefix_of_normalized_text(self):
        assert block_key("Enaku bayama iruku", length=4) == "enak"

    def test_shorter_than_length_returns_full_string(self):
        assert block_key("Hi", length=4) == "hi"

    def test_different_punctuation_same_key(self):
        assert block_key("enaku, bayama") == block_key("enaku bayama")


class TestSimilarityScore:
    def test_identical_strings_score_100(self):
        assert similarity_score("enaku bayama iruku", "enaku bayama iruku") == 100.0

    def test_completely_different_strings_score_low(self):
        score = similarity_score("enaku bayama iruku", "naan interview ku poren")
        assert score < 40.0

    def test_reordered_words_score_high(self):
        # token_sort_ratio (and the difflib fallback via normalized comparison)
        # should treat these as very similar despite word order.
        score = similarity_score("bayama enaku iruku", "enaku bayama iruku")
        assert score > 70.0

    def test_empty_strings_score_zero(self):
        assert similarity_score("", "enaku bayama") == 0.0
        assert similarity_score("enaku bayama", "") == 0.0
        assert similarity_score("", "") == 0.0

    def test_score_within_bounds(self):
        score = similarity_score("enaku konjam bayama iruku", "enaku bayama iruku")
        assert 0.0 <= score <= 100.0


def test_is_rapidfuzz_available_returns_bool():
    # We don't assert True/False since it depends on the environment;
    # we only assert the diagnostic surface behaves correctly.
    assert isinstance(is_rapidfuzz_available(), bool)
