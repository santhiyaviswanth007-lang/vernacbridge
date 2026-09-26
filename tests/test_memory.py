"""Tests for vernacbridge.memory (MemoryEngine, InMemoryStore, JSONFileStore)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vernacbridge.memory import InMemoryStore, JSONFileStore, MemoryEngine
from vernacbridge.protocols import MemoryEngineProtocol


def _turn(text: str) -> dict:
    """A minimal PipelineResult-shaped dict, enough for store-level tests."""
    return {
        "input": text,
        "normalized": text.lower(),
        "english_output": f"translated: {text}",
        "intent": "general",
        "emotion": "neutral",
        "keywords": [],
        "urgency": "low",
        "confidence": 90.0,
        "matched_sentence": text,
        "dataset_source": "sample",
        "is_confident": True,
        "top_matches": [],
        "session_id": None,
    }


class TestProtocolSatisfaction:
    def test_memory_engine_satisfies_protocol_structurally(self):
        assert isinstance(MemoryEngine(), MemoryEngineProtocol)


class TestInMemoryStore:
    def test_append_then_get_round_trips(self):
        store = InMemoryStore()
        store.append("s1", _turn("hello"))
        results = store.get("s1")
        assert len(results) == 1
        assert results[0]["input"] == "hello"

    def test_unknown_session_returns_empty_list(self):
        store = InMemoryStore()
        assert store.get("never-seen") == []

    def test_sessions_are_isolated(self):
        store = InMemoryStore()
        store.append("s1", _turn("a"))
        store.append("s2", _turn("b"))
        assert len(store.get("s1")) == 1
        assert len(store.get("s2")) == 1
        assert store.get("s1")[0]["input"] == "a"
        assert store.get("s2")[0]["input"] == "b"

    def test_order_is_oldest_first(self):
        store = InMemoryStore()
        store.append("s1", _turn("first"))
        store.append("s1", _turn("second"))
        store.append("s1", _turn("third"))
        assert [r["input"] for r in store.get("s1")] == ["first", "second", "third"]

    def test_capacity_evicts_oldest_first(self):
        store = InMemoryStore(capacity=2)
        store.append("s1", _turn("first"))
        store.append("s1", _turn("second"))
        store.append("s1", _turn("third"))
        results = store.get("s1")
        assert len(results) == 2
        assert [r["input"] for r in results] == ["second", "third"]

    def test_none_session_id_is_a_valid_key(self):
        store = InMemoryStore()
        store.append(None, _turn("no session"))
        assert len(store.get(None)) == 1


class TestJSONFileStore:
    def test_append_then_get_round_trips(self, tmp_path: Path):
        store = JSONFileStore(tmp_path / "history.jsonl")
        store.append("s1", _turn("hello"))
        results = store.get("s1")
        assert len(results) == 1
        assert results[0]["input"] == "hello"

    def test_get_before_any_write_returns_empty_list(self, tmp_path: Path):
        store = JSONFileStore(tmp_path / "does_not_exist_yet.jsonl")
        assert store.get("s1") == []

    def test_sessions_are_isolated(self, tmp_path: Path):
        store = JSONFileStore(tmp_path / "history.jsonl")
        store.append("s1", _turn("a"))
        store.append("s2", _turn("b"))
        assert len(store.get("s1")) == 1
        assert len(store.get("s2")) == 1

    def test_is_uncapped_unlike_in_memory_store(self, tmp_path: Path):
        store = JSONFileStore(tmp_path / "history.jsonl")
        for i in range(100):
            store.append("s1", _turn(f"turn-{i}"))
        assert len(store.get("s1")) == 100

    def test_persists_across_a_fresh_instance_pointed_at_the_same_file(self, tmp_path: Path):
        path = tmp_path / "history.jsonl"
        first = JSONFileStore(path)
        first.append("s1", _turn("before restart"))

        # A brand-new instance, simulating a process restart, must see it.
        second = JSONFileStore(path)
        results = second.get("s1")
        assert len(results) == 1
        assert results[0]["input"] == "before restart"

    def test_creates_parent_directories_if_missing(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "history.jsonl"
        store = JSONFileStore(nested)
        store.append("s1", _turn("x"))
        assert nested.exists()


class TestMemoryEngineDefaults:
    def test_default_short_term_is_an_in_memory_store(self):
        engine = MemoryEngine()
        assert isinstance(engine.short_term, InMemoryStore)

    def test_default_long_term_is_none(self):
        # Persistence across restarts is opt-in, not automatic.
        engine = MemoryEngine()
        assert engine.long_term is None

    def test_custom_short_term_store_is_used_instead_of_default(self):
        custom = InMemoryStore(capacity=1)
        engine = MemoryEngine(short_term=custom)
        assert engine.short_term is custom


class TestMemoryEngineRecordAndRecall:
    def test_record_writes_to_short_term(self):
        engine = MemoryEngine()
        engine.record("s1", _turn("hello"))
        assert len(engine.short_term.get("s1")) == 1

    def test_recall_without_long_term_reads_short_term(self):
        engine = MemoryEngine()
        engine.record("s1", _turn("hello"))
        assert engine.recall("s1") == engine.short_term.get("s1")

    def test_record_without_long_term_does_not_create_one(self):
        engine = MemoryEngine()
        engine.record("s1", _turn("hello"))
        assert engine.long_term is None  # still opt-in only

    def test_sessions_remain_isolated_through_the_engine(self):
        engine = MemoryEngine()
        engine.record("s1", _turn("a"))
        engine.record("s2", _turn("b"))
        assert len(engine.recall("s1")) == 1
        assert len(engine.recall("s2")) == 1


class TestMemoryEngineWithLongTerm:
    def test_record_writes_to_both_short_and_long_term(self, tmp_path: Path):
        long_term = JSONFileStore(tmp_path / "history.jsonl")
        engine = MemoryEngine(long_term=long_term)
        engine.record("s1", _turn("hello"))

        assert len(engine.short_term.get("s1")) == 1
        assert len(long_term.get("s1")) == 1

    def test_recall_prefers_long_term_when_present(self, tmp_path: Path):
        # Long-term is uncapped/authoritative; short-term (capacity=1) would
        # have evicted the first turn, but recall() must still return both.
        long_term = JSONFileStore(tmp_path / "history.jsonl")
        engine = MemoryEngine(short_term=InMemoryStore(capacity=1), long_term=long_term)
        engine.record("s1", _turn("first"))
        engine.record("s1", _turn("second"))

        assert len(engine.short_term.get("s1")) == 1  # capped, evicted "first"
        assert len(engine.recall("s1")) == 2  # long-term has both

    def test_long_term_persists_across_a_fresh_engine(self, tmp_path: Path):
        path = tmp_path / "history.jsonl"
        first_engine = MemoryEngine(long_term=JSONFileStore(path))
        first_engine.record("s1", _turn("before restart"))

        second_engine = MemoryEngine(long_term=JSONFileStore(path))
        results = second_engine.recall("s1")
        assert len(results) == 1
        assert results[0]["input"] == "before restart"


class TestMemoryEngineResultFidelity:
    def test_recall_returns_the_result_dict_unmodified(self):
        engine = MemoryEngine()
        original = _turn("hello")
        original["confidence"] = 87.5
        engine.record("s1", original)

        recalled = engine.recall("s1")[0]
        assert recalled["confidence"] == 87.5
        assert recalled["input"] == "hello"

    def test_recall_does_not_care_whether_explanation_key_is_present(self):
        # Phase 2's explain=True results carry an extra "explanation" key;
        # Phase 3 must not assume or require any particular result shape.
        engine = MemoryEngine()
        with_explanation = _turn("hello")
        with_explanation["explanation"] = {"chosen": {}, "candidate_count": 1}
        engine.record("s1", with_explanation)

        without_explanation = _turn("world")
        engine.record("s1", without_explanation)

        recalled = engine.recall("s1")
        assert "explanation" in recalled[0]
        assert "explanation" not in recalled[1]
