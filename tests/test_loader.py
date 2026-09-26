"""Tests for vernacbridge.loader.DatasetLoader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vernacbridge.exceptions import (
    DatasetDirectoryNotFoundError,
    DatasetReadError,
    EmptyDatasetError,
    NoDatasetsFoundError,
)
from vernacbridge.loader import DatasetLoader


class TestDiscoverCsvFiles:
    def test_finds_all_csv_files(self, valid_dataset_dir):
        loader = DatasetLoader(dataset_dir=valid_dataset_dir)
        files = loader.discover_csv_files()
        assert len(files) == 1
        assert files[0].name == "sample.csv"

    def test_finds_multiple_csv_files_sorted(self, messy_dataset_dir):
        loader = DatasetLoader(dataset_dir=messy_dataset_dir)
        files = loader.discover_csv_files()
        assert [f.name for f in files] == sorted(f.name for f in files)
        assert len(files) == 2

    def test_raises_when_directory_missing(self, tmp_path):
        loader = DatasetLoader(dataset_dir=tmp_path / "does_not_exist")
        with pytest.raises(DatasetDirectoryNotFoundError):
            loader.discover_csv_files()

    def test_raises_when_no_csv_files_present(self, empty_dataset_dir):
        loader = DatasetLoader(dataset_dir=empty_dataset_dir)
        with pytest.raises(NoDatasetsFoundError):
            loader.discover_csv_files()

    def test_excludes_master_dataset_csv_from_discovery(self, valid_dataset_dir, monkeypatch):
        # Simulate a previously-built master_dataset.csv sitting in the same dir.
        master_path = valid_dataset_dir / "master_dataset.csv"
        pd.DataFrame({"a": [1]}).to_csv(master_path, index=False)

        import vernacbridge.config as config

        monkeypatch.setattr(config, "DEFAULT_MASTER_DATASET_PATH", master_path)
        loader = DatasetLoader(dataset_dir=valid_dataset_dir)
        files = loader.discover_csv_files()
        assert master_path.resolve() not in {f.resolve() for f in files}


class TestLoadCsv:
    def test_loads_rows_and_tags_source_file(self, valid_dataset_dir):
        loader = DatasetLoader(dataset_dir=valid_dataset_dir)
        path = valid_dataset_dir / "sample.csv"
        df = loader.load_csv(path)
        assert (df["source_file"] == "sample.csv").all()
        assert len(df) == 3

    def test_raises_on_empty_csv(self, tmp_path: Path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        loader = DatasetLoader(dataset_dir=tmp_path)
        with pytest.raises(EmptyDatasetError):
            loader.load_csv(empty_csv)

    def test_raises_on_header_only_csv(self, tmp_path: Path):
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("tanglish_input,english_output,intent,emotion,keywords,urgency,confidence\n")
        loader = DatasetLoader(dataset_dir=tmp_path)
        with pytest.raises(EmptyDatasetError):
            loader.load_csv(header_only)

    def test_handles_latin1_encoded_file(self, tmp_path: Path):
        path = tmp_path / "latin1.csv"
        content = "tanglish_input,english_output,intent,emotion,keywords,urgency,confidence\n"
        content += "café pogalama,shall we go to the café,daily_conversation,happy,cafe,low,high\n"
        path.write_bytes(content.encode("latin-1"))
        loader = DatasetLoader(dataset_dir=tmp_path)
        df = loader.load_csv(path)  # should not raise despite non-utf-8 encoding
        assert len(df) == 1


class TestLoadAll:
    def test_loads_every_file_in_directory(self, messy_dataset_dir):
        loader = DatasetLoader(dataset_dir=messy_dataset_dir)
        frames = loader.load_all()
        assert len(frames) == 2
        total_rows = sum(len(f) for f in frames)
        assert total_rows == 8  # 3 in file_a + 5 in file_b

    def test_raises_when_all_files_fail_to_load(self, tmp_path: Path):
        bad = tmp_path / "bad.csv"
        bad.write_text("")
        loader = DatasetLoader(dataset_dir=tmp_path)
        with pytest.raises(NoDatasetsFoundError):
            loader.load_all()

    def test_skips_bad_file_but_loads_the_rest(self, tmp_path: Path, clean_rows):
        good = tmp_path / "good.csv"
        pd.DataFrame(clean_rows).to_csv(good, index=False)
        bad = tmp_path / "bad.csv"
        bad.write_text("")

        loader = DatasetLoader(dataset_dir=tmp_path)
        frames = loader.load_all()
        assert len(frames) == 1
        assert frames[0]["source_file"].iloc[0] == "good.csv"
