"""Tests for vernacbridge.cleaner.DatasetCleaner."""

from __future__ import annotations

import pandas as pd
import pytest

from vernacbridge.cleaner import DatasetCleaner


@pytest.fixture
def cleaner() -> DatasetCleaner:
    return DatasetCleaner(near_duplicate_threshold=92.0)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestExactDuplicates:
    def test_removes_byte_identical_rows(self, cleaner, clean_rows):
        rows = list(clean_rows) + [dict(clean_rows[0])]
        df, removed = cleaner.remove_exact_duplicates(_df(rows))
        assert removed == 1
        assert len(df) == len(clean_rows)

    def test_removes_case_and_punctuation_variant_duplicates(self, cleaner, clean_rows):
        variant = dict(clean_rows[0])
        variant["tanglish_input"] = "Enaku, Bayama Iruku!!"  # same normalized text
        df, removed = cleaner.remove_exact_duplicates(_df([clean_rows[0], variant]))
        assert removed == 1
        assert len(df) == 1

    def test_no_duplicates_removes_nothing(self, cleaner, clean_rows):
        df, removed = cleaner.remove_exact_duplicates(_df(clean_rows))
        assert removed == 0
        assert len(df) == len(clean_rows)


class TestNearDuplicates:
    def test_removes_near_duplicate_above_threshold(self, cleaner, clean_rows):
        near_dup = dict(clean_rows[0])
        near_dup["tanglish_input"] = "enaku bayama irukku"  # spelling variant, ~97% similar
        df, removed = cleaner.remove_near_duplicates(_df([clean_rows[0], near_dup]))
        assert removed == 1
        assert len(df) == 1

    def test_keeps_first_occurrence_of_a_cluster(self, cleaner, clean_rows):
        near_dup = dict(clean_rows[0])
        near_dup["tanglish_input"] = "enaku bayama irukku"
        df, removed = cleaner.remove_near_duplicates(_df([clean_rows[0], near_dup]))
        assert removed == 1
        assert len(df) == 1
        assert df.iloc[0]["tanglish_input"] == clean_rows[0]["tanglish_input"]

    def test_dissimilar_rows_both_kept(self, cleaner, clean_rows):
        df, removed = cleaner.remove_near_duplicates(_df(clean_rows))
        assert removed == 0
        assert len(df) == len(clean_rows)

    def test_semantically_different_sentence_not_treated_as_duplicate(self, cleaner, clean_rows):
        # Inserting a whole extra word ("konjam" = "a little") changes the
        # meaning of the sentence; it must NOT be flagged as a near-duplicate
        # even though it shares most of its characters with the original.
        different = dict(clean_rows[0])
        different["tanglish_input"] = "enaku konjam bayama iruku"
        df, removed = cleaner.remove_near_duplicates(_df([clean_rows[0], different]))
        assert removed == 0
        assert len(df) == 2

    def test_threshold_is_respected(self, clean_rows):
        near_dup = dict(clean_rows[0])
        near_dup["tanglish_input"] = "enaku bayama irukku"  # ~97% similar

        strict_cleaner = DatasetCleaner(near_duplicate_threshold=99.9)
        df, removed = strict_cleaner.remove_near_duplicates(_df([clean_rows[0], near_dup]))
        assert removed == 0  # 97% similar clears 92.0 but not a 99.9 threshold
        assert len(df) == 2


class TestCleanPipeline:
    def test_full_pipeline_removes_exact_and_near_duplicates(self, cleaner, clean_rows):
        exact_dup = dict(clean_rows[0])
        near_dup = dict(clean_rows[0])
        near_dup["tanglish_input"] = "enaku bayama irukku"

        rows = list(clean_rows) + [exact_dup, near_dup]
        cleaned, report = cleaner.clean(_df(rows))

        assert report.rows_in == len(rows)
        assert report.exact_duplicates_removed == 1
        assert report.near_duplicates_removed == 1
        assert report.rows_out == len(clean_rows)
        assert report.total_removed == 2

    def test_clean_resets_index(self, cleaner, clean_rows):
        cleaned, _ = cleaner.clean(_df(clean_rows))
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_empty_dataframe_returns_empty(self, cleaner):
        empty = pd.DataFrame(columns=["tanglish_input", "english_output"])
        cleaned, report = cleaner.clean(empty)
        assert cleaned.empty
        assert report.rows_in == 0
        assert report.rows_out == 0
