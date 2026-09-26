# Phase 1 — Translation Pipeline

**Status:** Complete. Built on the existing `MatchEngine`; no existing public
API changed.

## What this adds

A single structured "understanding" call for a Tanglish sentence:

```python
from vernacbridge import VernacBridge

vb = VernacBridge()
vb.translate_pipeline("enaku bayama iruku")
```

```json
{
  "input": "enaku bayama iruku",
  "normalized": "enaku bayama iruku",
  "english_output": "I am afraid.",
  "intent": "mental_health",
  "emotion": "fear",
  "keywords": ["fear", "afraid"],
  "urgency": "high",
  "confidence": 100.0,
  "matched_sentence": "enaku bayama iruku",
  "dataset_source": "general",
  "is_confident": true,
  "top_matches": []
}
```

Also available as `vernacbridge.translate_pipeline(...)` (module-level,
mirrors the existing `vernacbridge.translate()` convenience function) and as
the `TranslationPipeline` class directly for custom `MatchEngine` wiring.

## Why a new module instead of changing `Translator`

`Translator.translate()` already has a smaller, different return shape
(`translation`, `matched_input`, flat — no `keywords` list, no
`dataset_source`, no low-confidence candidate list) and existing tests and
callers depend on it. Changing its contract would violate backward
compatibility. `pipeline.py` is additive: a second, richer entry point for
callers (like a future LOVE AI integration) that need the full structured
result in one call, while `vernacbridge.translate()` keeps working exactly
as before for callers that don't.

## How it works

`TranslationPipeline.run()` calls `MatchEngine.top_matches()` **once** and
derives everything else from that single call:

```
normalize (utils.normalize_text)
        │
        ▼
MatchEngine.top_matches(text, n=3)   ← dataset search, one call
        │
        ▼
best = candidates[0]                  ← best-match selection
        │
        ├─ best.score >= threshold →  commit: english_output, intent,
        │                              emotion, urgency, confidence,
        │                              matched_sentence, dataset_source
        │                              all come directly off `best`
        │
        └─ best.score < threshold  →  is_confident=False; return
                                        candidates[:top_n] as top_matches
                                        instead of guessing
```

This is intentionally more efficient than composing today's `Translator` +
`EmotionDetector` + `IntentDetector` for the same text, which would run the
same fuzzy match three separate times — in this corpus-matching
architecture, intent and emotion are attributes of the matched row, not
independently computed, so one match answers all three questions.

## Design decisions worth knowing about later

- **`dataset_source` derivation.** The corpus stores the raw contributing
  filename (`source_file`, e.g. `"vernacbridge_career_1000.csv"`), but the
  spec's example shows a clean domain name (`"career"`). A small regex in
  `pipeline.py` (`_domain_from_source`) strips the `vernacbridge_` prefix
  and trailing `_<count>` suffix. It falls back to the bare filename stem
  for anything that doesn't match this convention, so it never raises on
  unexpected filenames (relevant once Phase 7's plugin-style dataset
  dropping arrives). If dataset filenames stop following this convention,
  this is the one place to update.
- **`keywords` as a list.** The CSV column is a comma-separated string;
  `_keywords_to_list()` splits and strips it, dropping empty tokens.
- **Low-confidence shape is uniform.** Rather than a different dict shape
  for the confident vs. unconfident case, both paths return the same keys
  (`is_confident` plus an always-present `top_matches`, empty when
  confident) so callers only need to branch on one boolean.
- **New config constant, not a new pattern.** `PIPELINE_TOP_N = 3` was added
  to `config.py` alongside `MATCH_CONFIDENCE_THRESHOLD`, following the
  project's existing single-source-of-truth convention for tunables rather
  than hardcoding `3` in `pipeline.py`.

## Extensibility (added after initial Phase 1 delivery)

`TranslationPipeline` is the single entry point every future
language-processing module plugs into: Context Engine (Phase 4), Memory
Engine (Phase 3), Plugin Manager (Phase 7), Conversation Engine (Phase 4 /
12). Each is accepted as an optional, keyword-only constructor argument and,
when present, is actually called at a defined point — see
`vernacbridge/protocols.py` for the four `Protocol` contracts and
`pipeline.py`'s module docstring for the exact call order and
error-handling rules (context/plugin failures propagate; memory/
conversation failures are logged and swallowed, since those are side
effects on an already-valid result).

With nothing injected — today's reality — every one of these steps is
skipped and behavior is unchanged from the original Phase 1 delivery.
`VernacBridge`'s constructor also accepts and forwards all four, so
`VernacBridge(memory_engine=..., context_engine=...)` wires them through
the shared facade engine.

Two bugs were found and fixed while wiring this up, worth knowing about if
this code is touched again: `_empty_result()` initially required a
`session_id` argument that `run()` didn't pass and didn't even accept as a
parameter, which crashed on any empty-input call; and the facade's
`VernacBridge.__init__`/`translate_pipeline()` hadn't been updated to
accept or forward the new constructor arguments at all, despite
`TranslationPipeline` itself already declaring them. Both are fixed and
covered by regression tests (`TestSessionIdThreading`,
`TestExtensionPointWiring`, `TestExtensionPointErrorHandling` in
`tests/test_pipeline.py`).

## Tests

`tests/test_pipeline.py` (39 tests) covers:
- `_domain_from_source` and `_keywords_to_list` helper correctness
- Confident-match structure and field correctness
- Low-confidence fallback: candidate count, ordering, per-candidate shape
- Empty/whitespace-only input handling
- Threshold parameter behavior (raising/lowering changes `is_confident`)
- `normalizer` injection
- `session_id` threading through all three result paths
- Each of the four extension-point hooks: called at the right point, with
  the right arguments, in the right order (`TestExtensionPointWiring`)
- Error isolation: context/plugin failures propagate, memory/conversation
  failures are logged and swallowed without losing a valid result
  (`TestExtensionPointErrorHandling`)

`tests/test_integration_real_dataset.py` gained 3 tests against the real
`datasets/master_dataset.csv` and the `VernacBridge` facade, including a
check that `vb.pipeline.engine is vb.engine` — i.e. the pipeline shares the
same loaded corpus as every other engine on the facade (this is the same
sharing behavior verified after the earlier `engine or MatchEngine()`
truthiness bugfix; `translate_pipeline` was written to receive an engine the
same way the fixed engines do, so it doesn't regress that fix).

Full suite: 130 tests, 129 passing. The one failure
(`test_reordered_words_score_high`) is a pre-existing, previously-diagnosed
difflib-fallback artifact unrelated to this phase — confirmed to pass with
RapidFuzz installed.
