# Phase 4 — Context Engine

**Status:** Complete. `pipeline.py` required **no changes** — Phase 1
already built the call site (`self.context_engine.resolve(raw_text,
session_id) or raw_text`, errors propagate) this phase plugs into.

## What this adds

```python
import pandas as pd
from vernacbridge import VernacBridge, MemoryEngine, ContextEngine, config

memory = MemoryEngine()
context = ContextEngine(memory_engine=memory)  # must share the SAME memory instance
vb = VernacBridge(memory_engine=memory, context_engine=context).load()

df = pd.read_csv(config.DEFAULT_MASTER_DATASET_PATH, dtype=str, keep_default_na=False)
first_sentence = df.iloc[500]["tanglish_input"]

vb.translate_pipeline(first_sentence, session_id="u1")
followup = vb.translate_pipeline("yean?", session_id="u1")  # "why?"

followup["matched_sentence"]  # resolves back to the same match as the first turn
```

With no `context_engine` injected — the default — every call is
independent, identical to Phases 1–3's behavior.

## Scope, deliberately limited

This is corpus-matching architecture, not a language model. A "real"
context engine could mean deep coreference resolution or topic tracking;
building that would mean inventing linguistic capability for Tanglish
specifically that hasn't been validated, and is out of V1's scope per the
project's mission boundary regardless.

What `ContextEngine.resolve()` actually does is mechanical: if the current
input is short (word count under `config.CONTEXT_FOLLOWUP_MAX_WORDS`,
default 3 — a *structural* signal, not a guess at specific Tanglish
question-words, which would risk encoding vocabulary assumptions this
project hasn't validated) and there's a confident prior turn on record for
this exact session, the prior turn's matched Tanglish corpus sentence is
prepended to the new text before it reaches `MatchEngine`. That's the
entire mechanism. No new capability is invented beyond string composition
over an existing memory read.

Long or complete-looking inputs, missing/unknown sessions, and unconfident
prior turns all pass through unchanged — `ContextEngine` never tries to be
clever about ambiguous cases; it only acts when the structural signal
(short input) and the data (a confident prior match) both clearly apply.

## Design decision: `memory_engine` is required, not optional

Every other `TranslationPipeline` collaborator (`plugin_manager`,
`conversation_engine`, and `memory_engine` itself) defaults to `None` and
is skipped when absent. `ContextEngine` does not follow that pattern, on
purpose.

The first bug ever found in this project was `self.engine = engine or
MatchEngine()` — a falsy check silently creating a disconnected instance
instead of sharing the intended one. `ContextEngine` is at direct risk of
the same failure mode: its entire purpose is reading memory that
`TranslationPipeline` writes to elsewhere. If it were allowed to
default-construct its own private `MemoryEngine()`, it would silently see
empty history forever, and the failure would look exactly like "context
resolution just doesn't work" with no obvious cause.

`ContextEngine(memory_engine=None)` raises `ValueError` immediately,
naming the fix in the error message. The same `MemoryEngine` instance must
be passed to both `ContextEngine(memory_engine=m)` and
`TranslationPipeline(memory_engine=m, context_engine=...)` /
`VernacBridge(memory_engine=m, context_engine=...)`.

This is verified two ways, not just documented:
- `test_facade_wiring_shares_memory_between_context_and_pipeline` confirms
  object identity (`is`) across all three points.
- `test_different_memory_instances_means_context_sees_nothing`
  deliberately wires two *different* instances and confirms resolution
  correctly does nothing, rather than silently misbehaving in a way that
  would be confusing to debug.

## How it integrates

```
raw_text
    │
    ▼
[NEW] ContextEngine.resolve(raw_text, session_id)
    │     - short input? confident prior turn in memory? -> fuse
    │     - otherwise -> unchanged
    ▼
(existing, unchanged) plugin pre-hook -> normalize -> MatchEngine.top_matches()
```

## Tests

`tests/test_context.py` (22 tests): required-dependency enforcement,
protocol satisfaction, every unchanged-passthrough case (empty input, long
input, missing session, unknown session, no prior turns, unconfident prior
turn), fusion correctness (using the Tanglish `matched_sentence`, never the
English output, since the fused text feeds back into Tanglish-to-Tanglish
matching), session isolation, recency window behavior (default window of 1
and a wider window of 2), configurable threshold behavior, the shared vs.
disconnected `MemoryEngine` scenarios above, full sequential pipeline
integration, and a real-corpus facade end-to-end test using an actual
confidently-matching row rather than a hand-picked example phrase.

Full suite at time of this phase: 209 tests, 208 passing. The one failure
(`test_reordered_words_score_high`) is the same pre-existing,
previously-diagnosed difflib-fallback artifact from earlier phases,
unrelated to Phase 4 — confirmed to pass with RapidFuzz installed.
