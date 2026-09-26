"""Tests for vernacbridge.context.ContextEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from vernacbridge.context import ContextEngine
from vernacbridge.core import MatchEngine
from vernacbridge.memory import MemoryEngine
from vernacbridge.pipeline import TranslationPipeline
from vernacbridge.protocols import ContextEngineProtocol


def _confident_result(matched_sentence: str) -> dict:
    """Minimal stand-in for a confident PipelineResult, for direct
    MemoryEngine.record() calls in tests that don't need a real pipeline."""
    return {
        "input": matched_sentence,
        "normalized": matched_sentence,
        "english_output": "some translation",
        "intent": "mental_health",
        "emotion": "fear",
        "keywords": [],
        "urgency": "high",
        "confidence": 100.0,
        "matched_sentence": matched_sentence,
        "dataset_source": "sample",
        "is_confident": True,
        "top_matches": [],
        "session_id": None,
    }


def _unconfident_result() -> dict:
    r = _confident_result("")
    r["is_confident"] = False
    r["matched_sentence"] = ""
    return r


@pytest.fixture
def memory() -> MemoryEngine:
    return MemoryEngine()


@pytest.fixture
def context(memory: MemoryEngine) -> ContextEngine:
    return ContextEngine(memory_engine=memory)


class TestRequiredMemoryEngine:
    def test_raises_without_a_memory_engine(self):
        with pytest.raises(ValueError, match="requires a memory_engine"):
            ContextEngine(memory_engine=None)

    def test_satisfies_context_engine_protocol(self, context):
        assert isinstance(context, ContextEngineProtocol)


class TestResolveUnchangedCases:
    def test_empty_text_returned_unchanged(self, context):
        assert context.resolve("", session_id="s1") == ""

    def test_long_input_returned_unchanged_even_with_history(self, memory, context):
        memory.record("s1", _confident_result("enaku bayama iruku"))
        long_text = "this is a complete standalone sentence with plenty of words"
        assert context.resolve(long_text, session_id="s1") == long_text

    def test_short_input_with_no_session_id_returned_unchanged(self, memory, context):
        memory.record("s1", _confident_result("enaku bayama iruku"))
        assert context.resolve("yean?", session_id=None) == "yean?"

    def test_short_input_with_unknown_session_returned_unchanged(self, memory, context):
        memory.record("s1", _confident_result("enaku bayama iruku"))
        assert context.resolve("yean?", session_id="never-seen") == "yean?"

    def test_short_input_with_no_prior_turns_returned_unchanged(self, context):
        assert context.resolve("yean?", session_id="s1") == "yean?"

    def test_short_input_after_unconfident_prior_turn_returned_unchanged(self, memory, context):
        memory.record("s1", _unconfident_result())
        assert context.resolve("yean?", session_id="s1") == "yean?"


class TestResolveFusionCases:
    def test_short_followup_is_fused_with_prior_confident_sentence(self, memory, context):
        memory.record("s1", _confident_result("enaku bayama iruku"))
        resolved = context.resolve("yean?", session_id="s1")
        assert resolved == "enaku bayama iruku yean?"

    def test_fusion_uses_matched_sentence_not_english_output(self, memory, context):
        # The fused text must stay in Tanglish, since it feeds back into
        # MatchEngine which matches against Tanglish corpus text.
        result = _confident_result("enaku bayama iruku")
        result["english_output"] = "I am afraid."
        memory.record("s1", result)
        resolved = context.resolve("yean?", session_id="s1")
        assert "I am afraid." not in resolved
        assert "enaku bayama iruku" in resolved

    def test_sessions_do_not_cross_contaminate(self, memory, context):
        memory.record("s1", _confident_result("enaku bayama iruku"))
        memory.record("s2", _confident_result("naan romba santhosham"))

        assert context.resolve("yean?", session_id="s1") == "enaku bayama iruku yean?"
        assert context.resolve("yean?", session_id="s2") == "naan romba santhosham yean?"


class TestRecencyWindow:
    def test_default_window_uses_only_the_most_recent_confident_turn(self, memory, context):
        memory.record("s1", _confident_result("first sentence"))
        memory.record("s1", _confident_result("second sentence"))
        resolved = context.resolve("yean?", session_id="s1")
        assert resolved == "second sentence yean?"
        assert "first sentence" not in resolved

    def test_wider_window_fuses_multiple_recent_turns(self, memory):
        wide_context = ContextEngine(memory_engine=memory, recency_window=2)
        memory.record("s1", _confident_result("first sentence"))
        memory.record("s1", _confident_result("second sentence"))
        resolved = wide_context.resolve("yean?", session_id="s1")
        assert resolved == "first sentence second sentence yean?"

    def test_wider_window_skips_unconfident_turns_within_the_window(self, memory):
        wide_context = ContextEngine(memory_engine=memory, recency_window=2)
        memory.record("s1", _unconfident_result())
        memory.record("s1", _confident_result("second sentence"))
        resolved = wide_context.resolve("yean?", session_id="s1")
        assert resolved == "second sentence yean?"


class TestConfigurableThreshold:
    def test_custom_max_followup_words_changes_what_counts_as_short(self, memory):
        memory.record("s1", _confident_result("enaku bayama iruku"))
        strict_context = ContextEngine(memory_engine=memory, max_followup_words=0)
        # "yean?" is 1 word -- exceeds a max of 0, so treated as complete.
        assert strict_context.resolve("yean?", session_id="s1") == "yean?"

    def test_default_threshold_comes_from_config(self, context):
        from vernacbridge import config

        assert context.max_followup_words == config.CONTEXT_FOLLOWUP_MAX_WORDS
        assert context.recency_window == config.CONTEXT_RECENCY_WINDOW


class TestSharedMemoryRequirement:
    """
    The scenario the required-memory_engine design exists to catch: two
    *different* MemoryEngine instances given to ContextEngine vs.
    TranslationPipeline. Resolution must silently do nothing (see nothing
    in its own, disconnected store) rather than crash -- but this test
    exists to make that failure mode visible and understood, not hidden.
    """

    def test_different_memory_instances_means_context_sees_nothing(self, master_csv: Path):
        pipeline_memory = MemoryEngine()
        context_memory = MemoryEngine()  # deliberately NOT the same instance
        context = ContextEngine(memory_engine=context_memory)

        pipeline = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            memory_engine=pipeline_memory,
            context_engine=context,
        )

        pipeline.run("enaku bayama iruku", session_id="s1")  # recorded into pipeline_memory
        result = pipeline.run("yean?", session_id="s1")  # context checks context_memory -- empty

        # context_memory never saw the first turn, so nothing was fused;
        # "yean?" alone gets matched (or fails to) on its own.
        assert result["input"] == "yean?"

    def test_shared_instance_correctly_resolves_the_followup(self, master_csv: Path):
        shared_memory = MemoryEngine()
        context = ContextEngine(memory_engine=shared_memory)

        pipeline = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            memory_engine=shared_memory,
            context_engine=context,
        )

        pipeline.run("enaku bayama iruku", session_id="s1")
        result = pipeline.run("yean?", session_id="s1")

        assert result["input"] == "enaku bayama iruku yean?"


class TestPipelineIntegration:
    def test_followup_resolves_to_the_same_match_as_the_original_turn(self, master_csv: Path):
        shared_memory = MemoryEngine()
        context = ContextEngine(memory_engine=shared_memory)
        pipeline = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            memory_engine=shared_memory,
            context_engine=context,
        )

        first = pipeline.run("enaku bayama iruku", session_id="s1")
        assert first["is_confident"] is True

        followup = pipeline.run("yean?", session_id="s1")
        assert followup["is_confident"] is True
        assert followup["matched_sentence"] == first["matched_sentence"]
        assert followup["english_output"] == first["english_output"]

    def test_context_engine_errors_propagate_not_swallowed(self, master_csv: Path):
        class RaisingContext:
            def resolve(self, text, session_id):
                raise RuntimeError("context backend unavailable")

        pipeline = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            context_engine=RaisingContext(),
        )
        with pytest.raises(RuntimeError, match="context backend unavailable"):
            pipeline.run("yean?", session_id="s1")


class TestFacadeIntegration:
    def test_facade_wiring_shares_memory_between_context_and_pipeline(self):
        from vernacbridge import ContextEngine as CE
        from vernacbridge import MemoryEngine as ME
        from vernacbridge import VernacBridge

        shared_memory = ME()
        context = CE(memory_engine=shared_memory)
        vb = VernacBridge(memory_engine=shared_memory, context_engine=context).load()

        assert vb.pipeline.context_engine is context
        assert vb.pipeline.memory_engine is shared_memory
        assert context.memory_engine is shared_memory

    def test_facade_end_to_end_followup_against_real_corpus(self):
        import pandas as pd

        from vernacbridge import ContextEngine as CE
        from vernacbridge import MemoryEngine as ME
        from vernacbridge import VernacBridge
        from vernacbridge import config as _config

        df = pd.read_csv(_config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
        sample_row = df.iloc[500]

        shared_memory = ME()
        context = CE(memory_engine=shared_memory)
        vb = VernacBridge(memory_engine=shared_memory, context_engine=context).load()

        first = vb.translate_pipeline(sample_row["tanglish_input"], session_id="u1")
        assert first["is_confident"] is True

        followup = vb.translate_pipeline("yean?", session_id="u1")
        assert followup["is_confident"] is True
        assert followup["english_output"] == sample_row["english_output"]
