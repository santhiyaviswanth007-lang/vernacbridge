# Phase 3 — Memory Layer

**Status:** Complete. `pipeline.py` required **no changes** — Phase 1 already
built the write hook this phase plugs into.

## What this adds

```python
from vernacbridge import VernacBridge, MemoryEngine

memory = MemoryEngine()
vb = VernacBridge(memory_engine=memory).load()

vb.translate_pipeline("enaku bayama iruku", session_id="u1")
vb.translate_pipeline("naan romba santhosham", session_id="u1")

memory.recall("u1")   # both turns, oldest first
```

With no `memory_engine` injected — the default — nothing is recorded
anywhere, and behavior is identical to every prior phase.

## Why this is the smallest-footprint phase so far

`TranslationPipeline._finish()` has called
`self.memory_engine.record(session_id, result)` (wrapped in a non-fatal
try/except) since **Phase 1** — written specifically so this phase could
plug a real implementation in without touching `pipeline.py` again. That
prediction held: Phase 3 adds one new file (`memory.py`), one new test
file, one config constant, and an export in `__init__.py`. Nothing else in
the codebase changed.

## Design decision: `MemoryEngine` is not auto-constructed

This directly parallels the correction made to Phase 2's design, applied
proactively before implementation rather than after:

Phase 2's `DatasetIntelligence` is auto-constructed by `TranslationPipeline`
because it's stateless and side-effect-free — calling it costs nothing and
touches no storage, so defaulting to a real instance was safe.
`MemoryEngine` is different: if present, it **writes data as a side effect
on every call**, at minimum to RAM, optionally to disk. Auto-constructing
one in `VernacBridge.__init__` the way `DatasetIntelligence` is would mean
every existing caller — including everyone already using
`vernacbridge.translate_pipeline()` — would silently start accumulating
conversation history without asking for it. That's a real behavior change
hiding behind an unchanged output shape.

So `memory_engine` stays `None` by default, exactly like `context_engine`,
`plugin_manager`, and `conversation_engine`. A caller must explicitly write
`VernacBridge(memory_engine=MemoryEngine())`. Long-term (disk) persistence
requires a second, separate opt-in beyond that — injecting a
`JSONFileStore` — so there are two deliberate gates, not one. Verified by
`test_no_memory_engine_injected_is_a_true_no_op` and the real-corpus check
that a plain `VernacBridge()` has `pipeline.memory_engine is None`.

## How "short-term / long-term / session" map onto the actual design

The original spec named three memory concerns. Rather than build three
separate subsystems, this phase models them as two swappable stores plus
one correlation key that already existed:

- **Short-term** — `InMemoryStore`, the default. Per-session, capped at
  `config.MEMORY_SHORT_TERM_CAPACITY` (50) via `collections.deque(maxlen=...)`,
  RAM-only, gone on process restart.
- **Long-term** — `JSONFileStore`, opt-in. Append-only JSON-lines file,
  uncapped, survives a restart (verified by
  `test_persists_across_a_fresh_instance_pointed_at_the_same_file` at the
  store level and `test_long_term_persists_across_a_fresh_engine` at the
  engine level).
- **Session** — not a physical store. `session_id` has been the
  correlation key threaded through the entire pipeline since Phase 1;
  every store here is keyed by it, so `recall(session_id)` *is*
  session-scoped memory by construction.

`MemoryEngine.record()` writes to `short_term` always, and additionally to
`long_term` if one was injected. `recall()` prefers `long_term` when
present (it's the uncapped, authoritative history) and falls back to
`short_term` otherwise — meaning a short `InMemoryStore` capacity doesn't
lose data as long as a `long_term` store is also configured (verified by
`test_recall_prefers_long_term_when_present`, which deliberately caps
short-term at 1 and confirms `recall()` still returns both turns).

## Why `TranslationPipeline.run()` doesn't call `recall()`

Reading memory back to resolve context — "Why?" becoming "Why am I
afraid?" — is Context Engine's job (Phase 4), via
`ContextEngineProtocol.resolve()`. `MemoryEngine` is independently
queryable today (call `.recall()` directly, as tests and any future
caller can), but nothing in the pipeline itself consumes it yet. This is
intentional: that wiring gets built once, when Context Engine actually
exists, rather than added now and reworked later.

## Where `MemoryStore` lives

`MemoryStore` (the `append`/`get` Protocol both `InMemoryStore` and
`JSONFileStore` satisfy) is defined in `memory.py`, not `protocols.py`.
`protocols.py` is scoped to contracts `TranslationPipeline` itself depends
on directly; `MemoryStore` is `MemoryEngine`'s own internal, swappable
collaborator, one layer removed from the pipeline. Keeping it local avoids
widening `protocols.py`'s stated purpose.

## Tests

`tests/test_memory.py` (25 tests): protocol satisfaction (`MemoryEngine`
structurally satisfies `MemoryEngineProtocol` via `isinstance`), both
stores individually (round-trip, session isolation, ordering, capacity
eviction, restart persistence, missing-parent-directory handling), and
`MemoryEngine` composing them (defaults, dual-write, recall preference,
and result fidelity — including a test confirming `recall()` doesn't care
whether a Phase 2 `"explanation"` key is present on a given turn, since
memory shouldn't couple to what produced the result it's storing).

`tests/test_pipeline.py` gained `TestRealMemoryEngineIntegration`: a real
(not stubbed) `MemoryEngine` accumulating actual turns through the real
pipeline, plus the explicit no-op regression guard for an uninjected
`memory_engine`.

`tests/test_integration_real_dataset.py` gained one test exercising a real
`MemoryEngine` against the actual 3,775-row corpus through the facade.

Full suite: **187 tests, 186 passing**. The one failure
(`test_reordered_words_score_high`) is the same pre-existing,
previously-diagnosed difflib-fallback artifact from earlier phases,
unrelated to Phase 3 — confirmed to pass with RapidFuzz installed.
