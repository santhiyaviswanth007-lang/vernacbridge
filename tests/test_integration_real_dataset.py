"""
Integration tests against the *real* generated ``datasets/master_dataset.csv``
and the top-level ``VernacBridge`` facade / module-level convenience
functions (``vernacbridge.translate``, etc.).

These are skipped automatically if the master dataset hasn't been built yet
(``DatasetEngine().build()`` must run first), so the unit tests in the rest
of the suite never depend on real data being present.
"""

from __future__ import annotations

import pytest

from vernacbridge import config

pytestmark = pytest.mark.skipif(
    not config.DEFAULT_MASTER_DATASET_PATH.exists(),
    reason="datasets/master_dataset.csv has not been built yet; run "
    "`python -m vernacbridge.dataset_engine` first.",
)


class TestRealMasterDataset:
    def test_master_dataset_loads_via_facade(self):
        from vernacbridge import VernacBridge

        vb = VernacBridge().load()
        assert len(vb.engine) > 0

    def test_translate_a_known_family_sentence(self):
        import pandas as pd

        from vernacbridge import VernacBridge

        df = pd.read_csv(config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
        sample_row = df.iloc[0]

        vb = VernacBridge()
        result = vb.translate(sample_row["tanglish_input"])
        # Translating a sentence taken verbatim from the corpus should be an
        # exact (or near-exact) match against itself.
        assert result["confidence"] >= 99.0
        assert result["translation"] == sample_row["english_output"]

    def test_search_by_intent_returns_only_that_intent(self):
        from vernacbridge import VernacBridge

        vb = VernacBridge()
        results = vb.search_by_intent("career", limit=10)
        assert results  # the career dataset is non-empty
        assert all(r["intent"] == "career" for r in results)

    def test_search_by_emotion_returns_only_that_emotion(self):
        from vernacbridge import VernacBridge

        vb = VernacBridge()
        results = vb.search_by_emotion("happy", limit=10)
        assert results
        assert all(r["emotion"] == "happy" for r in results)

    def test_module_level_convenience_functions_work(self):
        import vernacbridge

        result = vernacbridge.detect_emotion("enaku bayama iruku")
        assert result["emotion"] != "unknown" or result["confidence"] < config.MATCH_CONFIDENCE_THRESHOLD

    def test_translate_pipeline_on_a_known_sentence(self):
        import pandas as pd

        from vernacbridge import VernacBridge

        df = pd.read_csv(config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
        sample_row = df.iloc[100]

        vb = VernacBridge()
        result = vb.translate_pipeline(sample_row["tanglish_input"])

        assert result["is_confident"] is True
        assert result["confidence"] >= 99.0
        assert result["english_output"] == sample_row["english_output"]
        assert result["dataset_source"]  # non-empty; derived from source_file
        assert result["top_matches"] == []

    def test_translate_pipeline_shares_the_facade_engine(self):
        from vernacbridge import VernacBridge

        vb = VernacBridge().load()
        assert vb.pipeline.engine is vb.engine

    def test_module_level_translate_pipeline_works(self):
        import vernacbridge

        result = vernacbridge.translate_pipeline("enaku bayama iruku")
        assert "top_matches" in result
        assert "is_confident" in result

    def test_explain_mode_against_real_corpus(self):
        import pandas as pd

        from vernacbridge import VernacBridge

        df = pd.read_csv(config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
        sample_row = df.iloc[200]

        vb = VernacBridge()
        result = vb.translate_pipeline(sample_row["tanglish_input"], explain=True)

        assert result["is_confident"] is True
        assert "explanation" not in vb.translate_pipeline(sample_row["tanglish_input"])
        assert result["explanation"]["chosen"]["confidence"] == result["confidence"]
        assert result["explanation"]["candidate_count"] >= 1

    def test_memory_engine_against_real_corpus(self):
        import pandas as pd

        from vernacbridge import MemoryEngine, VernacBridge

        df = pd.read_csv(config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
        row_a = df.iloc[50]
        row_b = df.iloc[300]

        memory = MemoryEngine()
        vb = VernacBridge(memory_engine=memory)
        vb.translate_pipeline(row_a["tanglish_input"], session_id="workshop-session")
        vb.translate_pipeline(row_b["tanglish_input"], session_id="workshop-session")

        history = memory.recall("workshop-session")
        assert len(history) == 2
        assert history[0]["english_output"] == row_a["english_output"]
        assert history[1]["english_output"] == row_b["english_output"]
