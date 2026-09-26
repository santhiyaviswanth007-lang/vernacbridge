"""Tests for vernacbridge.validator.DatasetValidator."""

from __future__ import annotations

import pandas as pd
import pytest

from vernacbridge.exceptions import MissingColumnError
from vernacbridge.validator import DatasetValidator


@pytest.fixture
def validator() -> DatasetValidator:
    return DatasetValidator()


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestValidateColumns:
    def test_passes_with_all_required_columns(self, validator, clean_rows):
        validator.validate_columns(_df(clean_rows), "sample.csv")  # should not raise

    def test_raises_on_missing_column(self, validator):
        df = _df([{"tanglish_input": "hi", "english_output": "hi"}])
        with pytest.raises(MissingColumnError, match="sample.csv"):
            validator.validate_columns(df, "sample.csv")


class TestValidateRows:
    def test_all_valid_rows_pass_through_unchanged_count(self, validator, clean_rows):
        cleaned, report = validator.validate(_df(clean_rows), "sample.csv")
        assert report.rows_in == len(clean_rows)
        assert report.rows_out == len(clean_rows)
        assert report.rows_dropped == 0

    def test_drops_row_with_invalid_emotion(self, validator, clean_rows):
        rows = list(clean_rows)
        rows.append(
            {
                "tanglish_input": "office work over",
                "english_output": "Too much office work.",
                "intent": "career",
                "emotion": "furious",  # not in VALID_EMOTIONS
                "keywords": "work",
                "urgency": "medium",
                "confidence": "medium",
            }
        )
        cleaned, report = validator.validate(_df(rows), "sample.csv")
        assert report.dropped_invalid_emotion == 1
        assert "furious" in report.invalid_emotion_labels
        assert len(cleaned) == len(clean_rows)

    def test_drops_row_with_invalid_intent(self, validator, clean_rows):
        rows = list(clean_rows)
        rows.append(
            {
                "tanglish_input": "exam pathi worry",
                "english_output": "Worried about exam.",
                "intent": "studies",  # not in VALID_INTENTS
                "emotion": "anxious",
                "keywords": "exam",
                "urgency": "medium",
                "confidence": "medium",
            }
        )
        cleaned, report = validator.validate(_df(rows), "sample.csv")
        assert report.dropped_invalid_intent == 1
        assert "studies" in report.invalid_intent_labels
        assert len(cleaned) == len(clean_rows)

    def test_drops_row_with_empty_non_nullable_field(self, validator, clean_rows):
        rows = list(clean_rows)
        rows.append(
            {
                "tanglish_input": "idhu enna",
                "english_output": "   ",  # blank after strip
                "intent": "daily_conversation",
                "emotion": "confused",
                "keywords": "confused",
                "urgency": "low",
                "confidence": "low",
            }
        )
        cleaned, report = validator.validate(_df(rows), "sample.csv")
        assert report.dropped_empty == 1
        assert len(cleaned) == len(clean_rows)

    def test_normalizes_emotion_and_intent_case(self, validator, clean_rows):
        rows = [dict(clean_rows[0])]
        rows[0]["emotion"] = "  Fear "
        rows[0]["intent"] = "MENTAL_HEALTH"
        cleaned, report = validator.validate(_df(rows), "sample.csv")
        assert report.rows_dropped == 0
        assert cleaned.iloc[0]["emotion"] == "fear"
        assert cleaned.iloc[0]["intent"] == "mental_health"

    def test_multiple_invalid_rows_all_dropped_independently(self, validator, clean_rows):
        rows = list(clean_rows)
        rows.append({**clean_rows[0], "emotion": "furious"})
        rows.append({**clean_rows[0], "intent": "studies"})
        rows.append({**clean_rows[0], "english_output": ""})
        cleaned, report = validator.validate(_df(rows), "sample.csv")
        assert report.rows_dropped == 3
        assert len(cleaned) == len(clean_rows)
