"""
Shared pytest fixtures for the VernacBridge test suite.

Fixtures build small, deterministic datasets on disk (via ``tmp_path``)
rather than depending on the real, thousands-of-rows CSVs in ``datasets/``.
This keeps unit tests fast, isolated from data collection in progress, and
able to exercise specific edge cases (invalid labels, empty fields, exact
and near duplicates) on demand. A small number of integration tests in
``test_integration_real_dataset.py`` opt back into the real dataset files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vernacbridge import config

REQUIRED_COLUMNS = config.REQUIRED_COLUMNS


def _write_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def clean_rows() -> list[dict]:
    """A handful of valid, distinct rows spanning multiple intents/emotions."""
    return [
        {
            "tanglish_input": "enaku bayama iruku",
            "english_output": "I am afraid.",
            "intent": "mental_health",
            "emotion": "fear",
            "keywords": "fear,afraid",
            "urgency": "high",
            "confidence": "high",
        },
        {
            "tanglish_input": "naan romba santhosham ah irukken",
            "english_output": "I am very happy.",
            "intent": "daily_conversation",
            "emotion": "happy",
            "keywords": "happy,joy",
            "urgency": "low",
            "confidence": "high",
        },
        {
            "tanglish_input": "interview ku poren, romba nervous ah irukku",
            "english_output": "I am going for an interview, I am very nervous.",
            "intent": "career",
            "emotion": "anxious",
            "keywords": "interview,nervous",
            "urgency": "medium",
            "confidence": "high",
        },
    ]


@pytest.fixture
def dataset_dir_factory(tmp_path: Path):
    """Factory fixture: write an arbitrary set of CSVs into a fresh temp dir."""

    def _factory(files: dict[str, list[dict]]) -> Path:
        target = tmp_path / "datasets"
        target.mkdir(parents=True, exist_ok=True)
        for filename, rows in files.items():
            _write_csv(target / filename, rows)
        return target

    return _factory


@pytest.fixture
def valid_dataset_dir(dataset_dir_factory, clean_rows) -> Path:
    """A datasets/ directory with a single, fully valid CSV."""
    return dataset_dir_factory({"sample.csv": clean_rows})


@pytest.fixture
def messy_dataset_dir(dataset_dir_factory, clean_rows) -> Path:
    """
    A datasets/ directory spread across two files, engineered to exercise
    every validator/cleaner code path in one build:

    - an exact duplicate (row 0 of file_a repeated verbatim in file_b)
    - a near-duplicate (same meaning, minor wording change)
    - an invalid emotion label
    - an invalid intent label
    - an empty required field (english_output)
    """
    file_a = list(clean_rows)
    file_b = [
        dict(clean_rows[0]),  # exact duplicate of file_a row 0
        {
            "tanglish_input": "enaku bayama irukku",  # near-dup of row 0 (spelling variant)
            "english_output": "I am afraid.",
            "intent": "mental_health",
            "emotion": "fear",
            "keywords": "fear",
            "urgency": "medium",
            "confidence": "medium",
        },
        {
            "tanglish_input": "office la romba over work",
            "english_output": "Too much work at office.",
            "intent": "career",
            "emotion": "furious",  # invalid emotion label
            "keywords": "work",
            "urgency": "medium",
            "confidence": "medium",
        },
        {
            "tanglish_input": "exam pathi worry aagudhu",
            "english_output": "Worried about the exam.",
            "intent": "studies",  # invalid intent label
            "emotion": "anxious",
            "keywords": "exam",
            "urgency": "medium",
            "confidence": "medium",
        },
        {
            "tanglish_input": "idhu enna nadakudhu",
            "english_output": "",  # empty non-nullable field
            "intent": "daily_conversation",
            "emotion": "confused",
            "keywords": "confused",
            "urgency": "low",
            "confidence": "low",
        },
    ]
    return dataset_dir_factory({"file_a.csv": file_a, "file_b.csv": file_b})


@pytest.fixture
def missing_column_dataset_dir(tmp_path: Path) -> Path:
    """A datasets/ directory whose only CSV is missing a required column."""
    target = tmp_path / "datasets"
    target.mkdir(parents=True, exist_ok=True)
    bad_columns = [c for c in REQUIRED_COLUMNS if c != "keywords"]
    pd.DataFrame(
        [
            {
                "tanglish_input": "vanakkam",
                "english_output": "Hello.",
                "intent": "daily_conversation",
                "emotion": "happy",
                "urgency": "low",
                "confidence": "high",
            }
        ],
        columns=bad_columns,
    ).to_csv(target / "bad.csv", index=False)
    return target


@pytest.fixture
def empty_dataset_dir(tmp_path: Path) -> Path:
    """A datasets/ directory that exists but contains no CSV files."""
    target = tmp_path / "datasets"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def master_csv(tmp_path: Path, clean_rows: list[dict]) -> Path:
    """A ready-made master_dataset.csv for runtime-engine tests (core/translator/emotion/intent/search)."""
    path = tmp_path / "master_dataset.csv"
    rows = [dict(r, source_file="sample.csv") for r in clean_rows]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
