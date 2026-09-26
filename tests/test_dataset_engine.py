"""
Integration tests for vernacbridge.dataset_engine.DatasetEngine.

These exercise the full pipeline (load -> validate -> merge -> clean ->
write) end-to-end against small, disposable dataset directories, verifying
both the in-memory result and the on-disk artifacts (master_dataset.csv,
build_report.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vernacbridge.dataset_engine import DatasetEngine
from vernacbridge.exceptions import DatasetError


@pytest.fixture
def engine_paths(tmp_path: Path):
    return {
        "master": tmp_path / "out" / "master_dataset.csv",
        "report": tmp_path / "out" / "build_report.json",
    }


class TestDatasetEngineBuild:
    def test_builds_master_dataframe_from_valid_dataset(self, valid_dataset_dir, engine_paths, clean_rows):
        engine = DatasetEngine(
            dataset_dir=valid_dataset_dir,
            master_dataset_path=engine_paths["master"],
            build_report_path=engine_paths["report"],
        )
        df, report = engine.build()
        assert len(df) == len(clean_rows)
        assert report.total_rows_after_cleaning == len(clean_rows)
        assert report.files_loaded == ["sample.csv"]
        assert report.files_failed == []

    def test_messy_dataset_ends_up_deduplicated_and_validated(self, messy_dataset_dir, engine_paths, clean_rows):
        engine = DatasetEngine(
            dataset_dir=messy_dataset_dir,
            master_dataset_path=engine_paths["master"],
            build_report_path=engine_paths["report"],
        )
        df, report = engine.build()

        # 3 (file_a) + 5 (file_b) = 8 rows in; 1 empty-field + 1 invalid-emotion
        # + 1 invalid-intent dropped at validation = 5; then 1 exact dup + 1 near
        # dup removed at cleaning = 3 rows in the final master dataset.
        assert report.total_rows_before_validation == 8
        assert report.total_rows_after_validation == 5
        assert report.total_rows_after_cleaning == 3
        assert len(df) == 3

        cleaning = report.cleaning
        assert cleaning["exact_duplicates_removed"] == 1
        assert cleaning["near_duplicates_removed"] == 1

    def test_writes_master_csv_and_build_report_to_disk(self, valid_dataset_dir, engine_paths):
        engine = DatasetEngine(
            dataset_dir=valid_dataset_dir,
            master_dataset_path=engine_paths["master"],
            build_report_path=engine_paths["report"],
        )
        engine.build(write_output=True)

        assert engine_paths["master"].exists()
        assert engine_paths["report"].exists()

        on_disk = pd.read_csv(engine_paths["master"])
        assert len(on_disk) == 3
        assert set(on_disk.columns) >= {"tanglish_input", "english_output", "intent", "emotion"}

        report_json = json.loads(engine_paths["report"].read_text())
        assert report_json["total_rows_after_cleaning"] == 3

    def test_write_output_false_skips_disk_writes(self, valid_dataset_dir, engine_paths):
        engine = DatasetEngine(
            dataset_dir=valid_dataset_dir,
            master_dataset_path=engine_paths["master"],
            build_report_path=engine_paths["report"],
        )
        engine.build(write_output=False)
        assert not engine_paths["master"].exists()
        assert not engine_paths["report"].exists()

    def test_missing_column_file_is_skipped_not_fatal(
        self, dataset_dir_factory, clean_rows, missing_column_dataset_dir, engine_paths
    ):
        # Merge a valid file with the missing-column file in the same directory.
        import shutil

        combined_dir = missing_column_dataset_dir
        good_csv = combined_dir / "good.csv"
        pd.DataFrame(clean_rows).to_csv(good_csv, index=False)

        engine = DatasetEngine(
            dataset_dir=combined_dir,
            master_dataset_path=engine_paths["master"],
            build_report_path=engine_paths["report"],
        )
        df, report = engine.build()

        assert "bad.csv" in report.files_failed
        assert "good.csv" in report.files_loaded
        assert len(df) == len(clean_rows)

    def test_raises_when_no_dataset_survives_validation(self, dataset_dir_factory, engine_paths):
        all_invalid = [
            {
                "tanglish_input": "x",
                "english_output": "y",
                "intent": "not_a_real_intent",
                "emotion": "not_a_real_emotion",
                "keywords": "z",
                "urgency": "low",
                "confidence": "low",
            }
        ]
        dataset_dir = dataset_dir_factory({"all_bad.csv": all_invalid})
        engine = DatasetEngine(
            dataset_dir=dataset_dir,
            master_dataset_path=engine_paths["master"],
            build_report_path=engine_paths["report"],
        )
        with pytest.raises(DatasetError):
            engine.build()

    def test_adding_a_new_csv_requires_no_code_change(
        self, valid_dataset_dir, engine_paths, clean_rows
    ):
        """Regression test for the 'drop a CSV in, no code changes' requirement."""
        extra_rows = [
            {
                "tanglish_input": "veetuku poganum",
                "english_output": "I need to go home.",
                "intent": "family",
                "emotion": "calm",
                "keywords": "home",
                "urgency": "low",
                "confidence": "high",
            }
        ]
        pd.DataFrame(extra_rows).to_csv(valid_dataset_dir / "new_domain.csv", index=False)

        engine = DatasetEngine(
            dataset_dir=valid_dataset_dir,
            master_dataset_path=engine_paths["master"],
            build_report_path=engine_paths["report"],
        )
        df, report = engine.build()
        assert len(df) == len(clean_rows) + len(extra_rows)
        assert set(report.files_loaded) == {"sample.csv", "new_domain.csv"}
