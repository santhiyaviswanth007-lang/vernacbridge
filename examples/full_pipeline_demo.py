"""
VernacBridge — full pipeline demo.

Run this from the project root, after building the master dataset:

    python -m vernacbridge.dataset_engine   # only needed once, if
                                             # datasets/master_dataset.csv
                                             # doesn't exist yet
    python examples/full_pipeline_demo.py

Walks through everything V1 (Phases 1-4) can do today, in the order a real
caller would reach for it: basic translation, the full structured pipeline,
explain mode, and a two-turn conversation using Memory + Context together.

Every optional phase (2, 3, 4) is inert unless you explicitly opt into it --
this script shows both the "just translate" path and the fully-wired path,
so the difference is visible rather than assumed.
"""

from __future__ import annotations

import json

import pandas as pd

from vernacbridge import (
    VernacBridge,
    MemoryEngine,
    ContextEngine,
    config,
)


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # A real sentence from the actual corpus, so this demo reflects genuine
    # behavior rather than a hand-picked phrase that may not confidently match.
    df = pd.read_csv(config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
    sample = df.iloc[500]
    tanglish_sentence = sample["tanglish_input"]

    # --- Phase 1: basic translation, no extra setup -----------------------
    section("Phase 1 -- basic translate() (module-level convenience function)")
    import vernacbridge

    result = vernacbridge.translate(tanglish_sentence)
    print(f"Input:  {tanglish_sentence}")
    print(f"Output: {result['translation']}")
    print(f"Emotion / Intent: {result['emotion']} / {result['intent']}")

    # --- Phase 1: the full structured pipeline -----------------------------
    section("Phase 1 -- translate_pipeline() (structured result)")
    vb = VernacBridge().load()
    pipeline_result = vb.translate_pipeline(tanglish_sentence)
    print(json.dumps(pipeline_result, indent=2))

    # --- Phase 2: explain mode, opt-in only --------------------------------
    section("Phase 2 -- explain=True (why this match won)")
    explained = vb.translate_pipeline(tanglish_sentence, explain=True)
    print(json.dumps(explained["explanation"], indent=2))
    print()
    print("Note: a plain translate_pipeline() call above had NO 'explanation'")
    print("key at all -- explain mode changes nothing unless you ask for it.")

    # --- Phase 3 + 4: Memory and Context, wired together -------------------
    section("Phase 3 + 4 -- Memory and Context across a two-turn conversation")

    # ContextEngine requires the SAME MemoryEngine instance TranslationPipeline
    # uses -- passing two different instances would mean context resolution
    # silently sees an empty, disconnected history. See docs/PHASE_4_CONTEXT_ENGINE.md.
    memory = MemoryEngine()
    context = ContextEngine(memory_engine=memory)
    conversational_vb = VernacBridge(memory_engine=memory, context_engine=context).load()

    session_id = "demo-session"

    print(f"\nTurn 1: {tanglish_sentence!r}")
    turn_1 = conversational_vb.translate_pipeline(tanglish_sentence, session_id=session_id)
    print(f"  -> {turn_1['english_output']!r} (confident={turn_1['is_confident']})")

    print("\nTurn 2: 'yean?' (a bare follow-up -- Tanglish for 'why?')")
    turn_2 = conversational_vb.translate_pipeline("yean?", session_id=session_id)
    print(f"  -> {turn_2['english_output']!r} (confident={turn_2['is_confident']})")
    print("  ContextEngine resolved 'yean?' using turn 1's matched sentence")
    print(f"  before matching ran -- that's why it didn't just fail to match.")

    print(f"\nFull recorded history for session {session_id!r}:")
    history = memory.recall(session_id)
    for i, turn in enumerate(history, start=1):
        print(f"  {i}. input={turn['input']!r} -> {turn['english_output']!r}")


if __name__ == "__main__":
    main()
