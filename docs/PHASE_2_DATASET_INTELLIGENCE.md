# Phase 2 — Dataset Intelligence

**Status:** Complete. Built as an independent module over data
`TranslationPipeline` already computes; no Phase 1 file was redesigned,
only extended.

## What this adds

`vb.translate_pipeline(text)` — the default call — returns **exactly** the
same shape Phase 1 always returned. Nothing changed for existing callers.

Opting in with `explain=True` adds one extra key, `"explanation"`:

```python
from vernacbridge import VernacBridge

vb = VernacBridge()
result = vb.translate_pipeline("enaku bayama iruku", explain=True)
```

```json
{
  "...": "... all the usual Phase 1 fields, unchanged ...",
  "explanation": {
    "chosen": {
      "rank": 1,
      "tanglish_input": "enaku bayama iruku",
      "english_output": "I am afraid.",
      "intent": "mental_health",
      "emotion": "fear",
      "dataset_source": "general",
      "confidence": 100.0
    },
    "runner_up": { "...": "second-best candidate, same shape" },
    "margin": 56.25,
    "candidate_count": 3,
    "is_close_call": false,
    "ranking": ["...", "...", "..."]
  }
}
```

## Why explain mode is opt-in, not always-on

This was an explicit design correction: the first draft of this phase
would have made `DatasetIntelligence` an always-available collaborator
without changing the default output shape, which was correct as far as it
went — but didn't distinguish "library users who just want a translation"
from "developers debugging match quality." The approved design makes that
distinction structural, not just conventional:

- `PipelineResult`'s `"explanation"` key is declared `NotRequired` in the
  `TypedDict` — it's *absent* from the dict entirely when `explain=False`,
  not present-but-`None`. A caller who never asks for it sees zero schema
  drift.
- `DatasetIntelligence.explain()` is only ever called from one place in
  `pipeline.py`, guarded by `if explain:`. When `explain=False`, that call
  site is never reached — not a cheap no-op, an actual skip. This is what
  "zero performance impact when disabled" means concretely: verified by a
  test (`test_explain_false_never_calls_dataset_intelligence`) that injects
  a `DatasetIntelligence` double which raises if called, and confirms it's
  never triggered across the confident, low-confidence, and empty-input
  paths.

## Why `DatasetIntelligence` defaults to a real instance, unlike the other four

`TranslationPipeline` already had four optional collaborators from Phase 1
(`context_engine`, `memory_engine`, `plugin_manager`, `conversation_engine`),
all defaulting to `None` because their real implementations don't exist
yet. `DatasetIntelligence` breaks that pattern on purpose: Phase 2 *is* its
implementation, so `TranslationPipeline.__init__` default-constructs one
(`dataset_intelligence if dataset_intelligence is not None else
DatasetIntelligence()`) — the same lazy-default pattern already used for
`engine` (`MatchEngine`). This is what makes `explain=True` work with zero
extra wiring, while an alternate implementation can still be injected
(e.g. by a future Plugin System, Phase 7) without `pipeline.py` changing
again.

## How it integrates with the pipeline

No new matching happens. `DatasetIntelligence.explain()` receives the same
`candidates: list[Match]` that `MatchEngine.top_matches()` already produced
for the current query — pure analysis of a decision already made, called
once, right after the confident/low-confidence result is built and before
the plugin post-hook runs (so the explanation reflects the actual match
decision, not anything a plugin's `after_match()` later reshapes the
result into — verified by
`test_explanation_reflects_raw_candidates_not_plugin_mutated_result`).

```
normalize -> MatchEngine.top_matches()          <- unchanged Phase 1 core
        │
        ▼
build result (confident / low-confidence)         <- unchanged Phase 1 core
        │
        ▼
[NEW, only if explain=True] DatasetIntelligence.explain(candidates, top_n)
        │
        ▼
plugin post-hook -> memory -> conversation          <- unchanged Phase 1 order
```

## `close_call` semantics

A match is flagged `is_close_call: true` when the score margin between the
winner and the runner-up is below `config.CLOSE_CALL_MARGIN` (default
`10.0`). This is informational only — it does not affect `is_confident` or
which candidate is returned as the translation. It's meant to help a
developer notice "this answer was barely ahead of an equally plausible
alternative" even when it cleared the confidence threshold outright.

`close_call_margin` is also constructor-injectable on `DatasetIntelligence`
itself, for a caller who wants a different sensitivity without a
`config.py` edit.

## Tests

`tests/test_intelligence.py` (14 tests) covers the module in isolation:
empty/single/multiple-candidate explanation shapes, margin calculation,
ranking order and 1-indexing, `top_n` limiting, the close-call threshold
(both directions, plus confirming the default comes from `config.py`),
candidate field shape, JSON-serializability, and — via a `MatchEngine`
double that raises if called — confirmation that `explain()` never
triggers new matching.

`tests/test_pipeline.py` gained 13 tests across three new classes:
- `TestExplainModeDefaultBehaviorUnchanged` — the regression guard: no
  `"explanation"` key by default, identical key-set to Phase 1, and
  `DatasetIntelligence` never called when `explain=False`.
- `TestExplainModeEnabled` — the key is added and populated correctly on
  confident, low-confidence, and empty-input paths; default vs. injected
  `DatasetIntelligence` selection; explanation reflects raw match data
  even when a plugin mutates the visible result.
- `TestExplainModeFacadeIntegration` — `VernacBridge.translate_pipeline`
  forwards `explain` and `dataset_intelligence` correctly against real
  corpus data.

`tests/test_integration_real_dataset.py` gained one real-corpus explain
test.

Full suite: **158 tests, 157 passing**. The one failure
(`test_reordered_words_score_high`) is the same pre-existing,
previously-diagnosed difflib-fallback artifact from earlier phases,
unrelated to Phase 2 — confirmed to pass with RapidFuzz installed.
