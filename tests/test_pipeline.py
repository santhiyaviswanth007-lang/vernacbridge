"""Tests for vernacbridge.pipeline.TranslationPipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from vernacbridge.core import MatchEngine
from vernacbridge.pipeline import TranslationPipeline, _domain_from_source, _keywords_to_list


@pytest.fixture
def pipeline(master_csv: Path) -> TranslationPipeline:
    return TranslationPipeline(engine=MatchEngine(master_dataset_path=master_csv))


class TestDomainFromSource:
    def test_strips_vernacbridge_prefix_and_numeric_suffix(self):
        assert _domain_from_source("vernacbridge_career_1000.csv") == "career"

    def test_handles_multi_word_domain(self):
        assert _domain_from_source("vernacbridge_mental_health_500.csv") == "mental_health"

    def test_falls_back_to_stem_when_pattern_does_not_match(self):
        assert _domain_from_source("sample.csv") == "sample"

    def test_empty_source_returns_empty_string(self):
        assert _domain_from_source("") == ""


class TestKeywordsToList:
    def test_splits_and_strips_comma_separated_keywords(self):
        assert _keywords_to_list("fear, afraid ,scared") == ["fear", "afraid", "scared"]

    def test_empty_string_returns_empty_list(self):
        assert _keywords_to_list("") == []

    def test_drops_empty_tokens(self):
        assert _keywords_to_list("fear,,afraid,") == ["fear", "afraid"]


class TestConfidentMatch:
    def test_exact_match_returns_full_structured_result(self, pipeline):
        result = pipeline.run("enaku bayama iruku")

        assert result["is_confident"] is True
        assert result["english_output"] == "I am afraid."
        assert result["intent"] == "mental_health"
        assert result["emotion"] == "fear"
        assert result["keywords"] == ["fear", "afraid"]
        assert result["urgency"] == "high"
        assert result["matched_sentence"] == "enaku bayama iruku"
        assert result["dataset_source"] == "sample"
        assert result["confidence"] == 100.0
        assert result["top_matches"] == []

    def test_normalized_field_is_normalized_form_of_input(self, pipeline):
        result = pipeline.run("Enaku, Bayama Iruku!!")
        assert result["normalized"] == "enaku bayama iruku"
        assert result["input"] == "Enaku, Bayama Iruku!!"

    def test_confident_result_has_all_spec_keys(self, pipeline):
        result = pipeline.run("enaku bayama iruku")
        expected_keys = {
            "input", "normalized", "english_output", "intent", "emotion",
            "keywords", "urgency", "confidence", "matched_sentence",
            "dataset_source", "is_confident", "top_matches", "session_id",
        }
        assert set(result.keys()) == expected_keys


class TestLowConfidenceMatch:
    def test_below_threshold_returns_top_matches_not_a_translation(self, pipeline):
        result = pipeline.run("completely unrelated gibberish zzqx", min_confidence=60.0)

        assert result["is_confident"] is False
        assert result["english_output"] == ""
        assert result["intent"] == "unknown"
        assert result["emotion"] == "unknown"
        assert result["matched_sentence"] == ""
        assert result["dataset_source"] == ""
        assert len(result["top_matches"]) > 0

    def test_top_matches_are_sorted_best_first(self, pipeline):
        result = pipeline.run("completely unrelated gibberish zzqx", min_confidence=60.0)
        scores = [c["confidence"] for c in result["top_matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_matches_respects_top_n(self, pipeline):
        result = pipeline.run("completely unrelated gibberish zzqx", min_confidence=60.0, top_n=2)
        assert len(result["top_matches"]) <= 2

    def test_candidate_has_expected_shape(self, pipeline):
        result = pipeline.run("completely unrelated gibberish zzqx", min_confidence=60.0)
        candidate = result["top_matches"][0]
        expected_keys = {
            "tanglish_input", "english_output", "intent", "emotion",
            "keywords", "urgency", "confidence", "dataset_source",
        }
        assert set(candidate.keys()) == expected_keys
        assert isinstance(candidate["keywords"], list)


class TestEmptyInput:
    def test_empty_string_returns_unconfident_empty_result(self, pipeline):
        result = pipeline.run("")
        assert result["is_confident"] is False
        assert result["confidence"] == 0.0
        assert result["top_matches"] == []
        assert result["normalized"] == ""

    def test_whitespace_only_input_treated_as_empty(self, pipeline):
        result = pipeline.run("   ")
        assert result["is_confident"] is False
        assert result["normalized"] == ""

    def test_none_like_falsy_input_does_not_raise(self, pipeline):
        result = pipeline.run("")
        assert result["input"] == ""


class TestThresholdParameter:
    def test_lower_threshold_can_make_a_borderline_match_confident(self, pipeline):
        low = pipeline.run("completely unrelated gibberish zzqx", min_confidence=0.0)
        assert low["is_confident"] is True

    def test_higher_threshold_can_make_an_otherwise_confident_match_unconfident(self, pipeline):
        strict = pipeline.run("enaku bayama iruku", min_confidence=999.0)
        assert strict["is_confident"] is False
        assert len(strict["top_matches"]) > 0


class TestSessionIdThreading:
    def test_session_id_defaults_to_none(self, pipeline):
        assert pipeline.run("enaku bayama iruku")["session_id"] is None

    def test_session_id_is_echoed_back_on_confident_result(self, pipeline):
        result = pipeline.run("enaku bayama iruku", session_id="sess-1")
        assert result["session_id"] == "sess-1"

    def test_session_id_is_echoed_back_on_low_confidence_result(self, pipeline):
        result = pipeline.run("completely unrelated gibberish zzqx", session_id="sess-1")
        assert result["session_id"] == "sess-1"

    def test_session_id_is_echoed_back_on_empty_input(self, pipeline):
        assert pipeline.run("", session_id="sess-1")["session_id"] == "sess-1"


class TestDependencyDefaultsAndStorage:
    def test_future_dependencies_default_to_none(self, master_csv):
        p = TranslationPipeline(engine=MatchEngine(master_dataset_path=master_csv))
        assert p.context_engine is None
        assert p.memory_engine is None
        assert p.plugin_manager is None
        assert p.conversation_engine is None

    def test_no_dependencies_injected_behaves_exactly_as_before(self, master_csv):
        # The regression guard: with nothing injected, every extension
        # point is skipped and output is unaffected by their existence.
        engine = MatchEngine(master_dataset_path=master_csv)
        plain = TranslationPipeline(engine=engine)
        explicit_none = TranslationPipeline(
            engine=engine,
            context_engine=None,
            memory_engine=None,
            plugin_manager=None,
            conversation_engine=None,
        )
        assert plain.run("enaku bayama iruku") == explicit_none.run("enaku bayama iruku")

    def test_constructor_signature_is_keyword_only_for_future_deps(self, master_csv):
        # Positional calls using only the current, stable dependencies
        # must keep working untouched -- this is the backward-compatibility
        # guarantee the keyword-only design is meant to protect.
        engine = MatchEngine(master_dataset_path=master_csv)
        p = TranslationPipeline(engine)  # positional, as existing code does
        assert p.engine is engine


class _RecordingContextEngine:
    """Test double satisfying ContextEngineProtocol."""

    def __init__(self, rewrite: str | None = None):
        self.calls: list[tuple[str, str | None]] = []
        self.rewrite = rewrite

    def resolve(self, text: str, session_id: str | None) -> str:
        self.calls.append((text, session_id))
        return self.rewrite if self.rewrite is not None else text


class _RecordingMemoryEngine:
    """Test double satisfying MemoryEngineProtocol."""

    def __init__(self):
        self.recorded: list[tuple[str | None, dict]] = []

    def record(self, session_id, result) -> None:
        self.recorded.append((session_id, result))

    def recall(self, session_id):
        return [r for sid, r in self.recorded if sid == session_id]


class _RecordingPluginManager:
    """Test double satisfying PluginManagerProtocol."""

    def __init__(self, rewrite: str | None = None):
        self.before_calls: list[str] = []
        self.after_calls: list[dict] = []
        self.rewrite = rewrite

    def before_match(self, text: str) -> str:
        self.before_calls.append(text)
        return self.rewrite if self.rewrite is not None else text

    def after_match(self, result: dict) -> dict:
        self.after_calls.append(result)
        return {**result, "plugin_touched": True}


class _RecordingConversationEngine:
    """Test double satisfying ConversationEngineProtocol."""

    def __init__(self):
        self.calls: list[tuple[str | None, dict]] = []

    def next_turn(self, session_id, result) -> None:
        self.calls.append((session_id, result))


class _RaisingMemoryEngine:
    def record(self, session_id, result) -> None:
        raise RuntimeError("memory backend unavailable")


class _RaisingConversationEngine:
    def next_turn(self, session_id, result) -> None:
        raise RuntimeError("conversation engine unavailable")


class _RaisingContextEngine:
    def resolve(self, text, session_id) -> str:
        raise RuntimeError("context backend unavailable")


class TestExtensionPointWiring:
    """
    Verifies each of the four extension points is actually called, at the
    right point, with the right arguments -- not just accepted and stored.
    """

    def _make(self, master_csv, **kwargs):
        return TranslationPipeline(engine=MatchEngine(master_dataset_path=master_csv), **kwargs)

    def test_context_engine_resolve_is_called_before_matching(self, master_csv):
        ctx = _RecordingContextEngine(rewrite="enaku bayama iruku")
        p = self._make(master_csv, context_engine=ctx)
        result = p.run("why?", session_id="sess-1")

        assert ctx.calls == [("why?", "sess-1")]
        # The rewritten text, not the original, is what got matched.
        assert result["is_confident"] is True
        assert result["matched_sentence"] == "enaku bayama iruku"

    def test_plugin_before_match_can_rewrite_the_query(self, master_csv):
        plugin = _RecordingPluginManager(rewrite="enaku bayama iruku")
        p = self._make(master_csv, plugin_manager=plugin)
        result = p.run("gibberish", session_id="sess-1")

        assert plugin.before_calls == ["gibberish"]
        assert result["matched_sentence"] == "enaku bayama iruku"

    def test_plugin_after_match_can_modify_the_result(self, master_csv):
        plugin = _RecordingPluginManager()
        p = self._make(master_csv, plugin_manager=plugin)
        result = p.run("enaku bayama iruku")

        assert len(plugin.after_calls) == 1
        assert result["plugin_touched"] is True

    def test_memory_engine_records_after_result_is_built(self, master_csv):
        memory = _RecordingMemoryEngine()
        p = self._make(master_csv, memory_engine=memory)
        result = p.run("enaku bayama iruku", session_id="sess-1")

        assert len(memory.recorded) == 1
        recorded_session, recorded_result = memory.recorded[0]
        assert recorded_session == "sess-1"
        assert recorded_result == result

    def test_conversation_engine_notified_last(self, master_csv):
        order: list[str] = []

        class OrderingMemory(_RecordingMemoryEngine):
            def record(self, session_id, result):
                order.append("memory")
                super().record(session_id, result)

        class OrderingConversation(_RecordingConversationEngine):
            def next_turn(self, session_id, result):
                order.append("conversation")
                super().next_turn(session_id, result)

        p = self._make(
            master_csv,
            memory_engine=OrderingMemory(),
            conversation_engine=OrderingConversation(),
        )
        p.run("enaku bayama iruku")

        assert order == ["memory", "conversation"]

    def test_full_call_order_across_all_four_hooks(self, master_csv):
        order: list[str] = []

        class OrderedContext(_RecordingContextEngine):
            def resolve(self, text, session_id):
                order.append("context")
                return super().resolve(text, session_id)

        class OrderedPlugin(_RecordingPluginManager):
            def before_match(self, text):
                order.append("plugin_before")
                return super().before_match(text)

            def after_match(self, result):
                order.append("plugin_after")
                return super().after_match(result)

        class OrderedMemory(_RecordingMemoryEngine):
            def record(self, session_id, result):
                order.append("memory")
                super().record(session_id, result)

        class OrderedConversation(_RecordingConversationEngine):
            def next_turn(self, session_id, result):
                order.append("conversation")
                super().next_turn(session_id, result)

        p = self._make(
            master_csv,
            context_engine=OrderedContext(),
            plugin_manager=OrderedPlugin(),
            memory_engine=OrderedMemory(),
            conversation_engine=OrderedConversation(),
        )
        p.run("enaku bayama iruku")

        assert order == ["context", "plugin_before", "plugin_after", "memory", "conversation"]


class TestExtensionPointErrorHandling:
    """
    Error handling is intentionally asymmetric: context/plugin hooks can
    change what gets matched or returned, so their failures propagate.
    Memory/conversation hooks are side effects on an already-valid result,
    so their failures are logged and swallowed.
    """

    def _make(self, master_csv, **kwargs):
        return TranslationPipeline(engine=MatchEngine(master_dataset_path=master_csv), **kwargs)

    def test_context_engine_failure_propagates(self, master_csv):
        p = self._make(master_csv, context_engine=_RaisingContextEngine())
        with pytest.raises(RuntimeError, match="context backend unavailable"):
            p.run("enaku bayama iruku")

    def test_memory_engine_failure_does_not_break_the_pipeline(self, master_csv):
        p = self._make(master_csv, memory_engine=_RaisingMemoryEngine())
        result = p.run("enaku bayama iruku")  # must not raise
        assert result["is_confident"] is True
        assert result["english_output"] == "I am afraid."

    def test_conversation_engine_failure_does_not_break_the_pipeline(self, master_csv):
        p = self._make(master_csv, conversation_engine=_RaisingConversationEngine())
        result = p.run("enaku bayama iruku")  # must not raise
        assert result["is_confident"] is True

    def test_memory_and_conversation_failures_are_independent(self, master_csv):
        # A broken memory engine must not prevent the (also present)
        # conversation engine from still being notified.
        conversation = _RecordingConversationEngine()
        p = self._make(
            master_csv,
            memory_engine=_RaisingMemoryEngine(),
            conversation_engine=conversation,
        )
        p.run("enaku bayama iruku")
        assert len(conversation.calls) == 1


class TestNormalizerInjection:
    def test_default_normalizer_is_normalize_text(self, master_csv):
        p = TranslationPipeline(engine=MatchEngine(master_dataset_path=master_csv))
        assert p.normalizer("Enaku, Bayama Iruku!!") == "enaku bayama iruku"

    def test_custom_normalizer_is_used_instead_of_default(self, master_csv):
        calls: list[str] = []

        def spy_normalizer(text: str) -> str:
            calls.append(text)
            return text.lower()

        p = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            normalizer=spy_normalizer,
        )
        result = p.run("ENAKU BAYAMA IRUKU")

        assert calls == ["ENAKU BAYAMA IRUKU"]
        assert result["normalized"] == "enaku bayama iruku"

    def test_custom_normalizer_that_returns_empty_string_short_circuits(self, master_csv):
        p = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            normalizer=lambda text: "",
        )
        result = p.run("enaku bayama iruku")
        assert result["is_confident"] is False
        assert result["normalized"] == ""


class TestExplainModeDefaultBehaviorUnchanged:
    """
    Phase 2 regression guard: explain=False (the default) must produce
    output identical in shape and content to Phase 1 -- no extra key, no
    DatasetIntelligence call, no behavior change for existing callers.
    """

    def test_default_call_has_no_explanation_key(self, pipeline):
        result = pipeline.run("enaku bayama iruku")
        assert "explanation" not in result

    def test_explicit_explain_false_has_no_explanation_key(self, pipeline):
        result = pipeline.run("enaku bayama iruku", explain=False)
        assert "explanation" not in result

    def test_confident_result_key_set_is_unchanged_by_explain_param_existing(self, pipeline):
        # Same assertion as TestConfidentMatch.test_confident_result_has_all_spec_keys,
        # kept here too as an explicit "Phase 2 didn't change the default shape" guard.
        result = pipeline.run("enaku bayama iruku")
        expected_keys = {
            "input", "normalized", "english_output", "intent", "emotion",
            "keywords", "urgency", "confidence", "matched_sentence",
            "dataset_source", "is_confident", "top_matches", "session_id",
        }
        assert set(result.keys()) == expected_keys

    def test_explain_false_never_calls_dataset_intelligence(self, master_csv):
        class ExplodingIntelligence:
            def explain(self, candidates, top_n):
                raise AssertionError("DatasetIntelligence must not be called when explain=False")

        p = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            dataset_intelligence=ExplodingIntelligence(),
        )
        p.run("enaku bayama iruku")  # must not raise
        p.run("")  # empty-input path too
        p.run("completely unrelated gibberish zzqx")  # low-confidence path too


class TestExplainModeEnabled:
    def test_explain_true_adds_explanation_key_on_confident_match(self, pipeline):
        result = pipeline.run("enaku bayama iruku", explain=True)
        assert "explanation" in result
        assert result["explanation"] is not None
        assert result["explanation"]["chosen"]["tanglish_input"] == "enaku bayama iruku"

    def test_explain_true_adds_explanation_key_on_low_confidence_match(self, pipeline):
        result = pipeline.run("completely unrelated gibberish zzqx", explain=True)
        assert "explanation" in result
        assert result["explanation"] is not None
        assert result["explanation"]["candidate_count"] > 0

    def test_explain_true_on_empty_input_sets_explanation_to_none(self, pipeline):
        result = pipeline.run("", explain=True)
        assert "explanation" in result
        assert result["explanation"] is None

    def test_default_dataset_intelligence_is_used_when_none_injected(self, master_csv):
        from vernacbridge.intelligence import DatasetIntelligence

        p = TranslationPipeline(engine=MatchEngine(master_dataset_path=master_csv))
        assert isinstance(p.dataset_intelligence, DatasetIntelligence)

    def test_custom_dataset_intelligence_is_used_instead_of_default(self, master_csv):
        class StubIntelligence:
            def __init__(self):
                self.calls = []

            def explain(self, candidates, top_n):
                self.calls.append((len(candidates), top_n))
                return {"stub": True}

        stub = StubIntelligence()
        p = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            dataset_intelligence=stub,
        )
        result = p.run("enaku bayama iruku", explain=True, top_n=2)

        assert result["explanation"] == {"stub": True}
        assert stub.calls == [(2, 2)]  # top_n=2 limits candidates fetched from the engine to 2

    def test_explanation_reflects_raw_candidates_not_plugin_mutated_result(self, master_csv):
        # The explanation must describe the actual match decision, not
        # whatever a plugin's after_match() later reshapes the result into.
        class MutatingPlugin:
            def before_match(self, text):
                return text

            def after_match(self, result):
                return {**result, "english_output": "PLUGIN OVERRIDE"}

        p = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            plugin_manager=MutatingPlugin(),
        )
        result = p.run("enaku bayama iruku", explain=True)

        assert result["english_output"] == "PLUGIN OVERRIDE"
        assert result["explanation"]["chosen"]["english_output"] == "I am afraid."


class TestExplainModeFacadeIntegration:
    def test_facade_forwards_explain_true(self):
        import pandas as pd

        from vernacbridge import VernacBridge
        from vernacbridge import config as _config

        df = pd.read_csv(_config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
        sample_row = df.iloc[100]

        vb = VernacBridge()
        result = vb.translate_pipeline(sample_row["tanglish_input"], explain=True)

        assert "explanation" in result
        assert result["explanation"]["chosen"]["english_output"] == sample_row["english_output"]

    def test_facade_default_call_has_no_explanation_key(self):
        from vernacbridge import VernacBridge

        vb = VernacBridge()
        result = vb.translate_pipeline("enaku bayama iruku")
        assert "explanation" not in result

    def test_facade_accepts_custom_dataset_intelligence(self):
        from vernacbridge import VernacBridge

        class StubIntelligence:
            def explain(self, candidates, top_n):
                return {"stub": True}

        vb = VernacBridge(dataset_intelligence=StubIntelligence())
        result = vb.translate_pipeline("enaku bayama iruku", explain=True)
        assert result["explanation"] == {"stub": True}


class TestRealMemoryEngineIntegration:
    """
    Phase 1 already verified the memory_engine hook mechanics with a stub
    (TestExtensionPointWiring). These tests confirm a real MemoryEngine
    (Phase 3) actually accumulates genuine pipeline turns through that same
    hook, and that its absence is truly a no-op.
    """

    def test_no_memory_engine_injected_is_a_true_no_op(self, master_csv):
        # The regression guard for the Phase 3 design decision: a pipeline
        # with no memory_engine must behave identically to one before
        # MemoryEngine existed at all -- nothing recorded, nothing to leak.
        p = TranslationPipeline(engine=MatchEngine(master_dataset_path=master_csv))
        assert p.memory_engine is None
        result = p.run("enaku bayama iruku", session_id="s1")
        assert result["is_confident"] is True  # unaffected either way

    def test_real_memory_engine_accumulates_turns_through_the_pipeline(self, master_csv):
        from vernacbridge.memory import MemoryEngine

        memory = MemoryEngine()
        p = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            memory_engine=memory,
        )
        p.run("enaku bayama iruku", session_id="s1")
        p.run("naan romba santhosham ah irukken", session_id="s1")
        p.run("interview ku poren, romba nervous ah irukku", session_id="s2")

        s1_history = memory.recall("s1")
        s2_history = memory.recall("s2")
        assert len(s1_history) == 2
        assert len(s2_history) == 1
        assert s1_history[0]["input"] == "enaku bayama iruku"
        assert s1_history[1]["input"] == "naan romba santhosham ah irukken"

    def test_recorded_result_matches_what_the_pipeline_returned(self, master_csv):
        from vernacbridge.memory import MemoryEngine

        memory = MemoryEngine()
        p = TranslationPipeline(
            engine=MatchEngine(master_dataset_path=master_csv),
            memory_engine=memory,
        )
        returned = p.run("enaku bayama iruku", session_id="s1", explain=True)
        recorded = memory.recall("s1")[0]

        assert recorded == returned
        assert "explanation" in recorded  # explain=True result, faithfully stored
