# VernacBridge

A Python library that helps Tamil-speaking people express what they mean in
**Tanglish** (Tamil written in English letters) and get it back in clear
English — with intent, emotion, and confidence preserved, not just a literal
word-for-word translation.

The goal isn't just "translate text." It's to help someone who knows
exactly what they want to say, but feels less confident saying it in
English, communicate it anyway — for messages, notes, and everyday
technical or personal communication — without losing what they meant or
how they felt saying it.

## Status — V1, Phases 1–4 complete

| Phase | What it adds |
|---|---|
| 1 — Translation Pipeline | Normalize → match against the corpus → structured result: translation, intent, emotion, keywords, urgency, confidence. Honest about uncertainty: below-threshold matches return the closest candidates instead of a guess. |
| 2 — Dataset Intelligence | Opt-in (`explain=True`) transparency into *why* a match won: runner-up, score margin, full ranking. Zero effect on the default output or performance when unused. |
| 3 — Memory Layer | Opt-in, session-scoped conversation history (`MemoryEngine`), short-term (RAM) and long-term (disk) storage. Nothing is recorded unless you explicitly inject it. |
| 4 — Context Engine | Resolves short follow-ups ("Why?") against the most recent confident turn in the same session, using Memory. Mechanical (word-count + prior-turn lookup), not a language model. |

Verified against a real, human-labelled ~4,000-row Tanglish corpus spanning
career, education, family, and general-conversation domains. 209 automated
tests.

Every phase beyond 1 is **inert by default** — a caller who just does
`vernacbridge.translate(...)` gets exactly Phase 1's behavior. Nothing is
enabled, recorded, or computed unless you ask for it.

## Installation

```bash
git clone https://github.com/santhiyaviswanth007-lang/vernacbridge.git
cd vernacbridge
pip install .
```

That's a genuine, non-editable install — `datasets/master_dataset.csv` ships
as real package data, so `import vernacbridge` works from anywhere, not
just from inside this repo.

For development (editable install, plus tests and linting):

```bash
pip install -e ".[dev]"
```

`requirements.txt` lists the same runtime dependencies directly, if you'd
rather manage them yourself: `pip install -r requirements.txt`.

RapidFuzz is used for similarity scoring; if it isn't installed, VernacBridge
falls back to `difflib` with a logged warning (lower accuracy, same API).

## Quick start

```python
import vernacbridge

vernacbridge.translate("enaku bayama iruku")
vernacbridge.detect_emotion("enaku bayama iruku")
vernacbridge.detect_intent("enaku bayama iruku")
vernacbridge.search_by_emotion("fear")
```

### The full structured pipeline

```python
vernacbridge.translate_pipeline("enaku bayama iruku")
```

Returns translation, intent, emotion, keywords, urgency, confidence, the
matched corpus sentence, and — if confidence is too low to commit to one
answer — the closest candidates instead of a wrong guess.

### Explain mode (Phase 2) — why a match won

```python
vernacbridge.translate_pipeline("enaku bayama iruku", explain=True)
```

Adds an `"explanation"` key: the runner-up candidate, the score margin, and
whether it was a close call. Absent entirely unless you ask for it.

### Memory and Context (Phases 3–4) — conversation continuity

```python
from vernacbridge import VernacBridge, MemoryEngine, ContextEngine

memory = MemoryEngine()
context = ContextEngine(memory_engine=memory)  # must share the SAME memory instance
vb = VernacBridge(memory_engine=memory, context_engine=context).load()

vb.translate_pipeline("enaku bayama iruku", session_id="user-1")
vb.translate_pipeline("yean?", session_id="user-1")  # "why?" -- resolved using the prior turn

memory.recall("user-1")  # full turn history for this session
```

`ContextEngine` requires `memory_engine` explicitly, and it must be the
*same instance* given to `VernacBridge`/`TranslationPipeline` — this is
enforced (it raises if omitted), not just documented, so context resolution
can't silently read from an empty, disconnected memory store.

### Custom dataset path / multiple independent instances

```python
from vernacbridge import VernacBridge

vb = VernacBridge(master_dataset_path="path/to/master_dataset.csv").load()
vb.translate("enaku bayama iruku")
```

See `examples/full_pipeline_demo.py` for all of the above run together
against the real corpus.

## Building the master dataset

Raw, per-domain CSVs live in `vernacbridge/datasets/` (columns:
`tanglish_input`, `english_output`, `intent`, `emotion`, `keywords`,
`urgency`, `confidence`). Adding a new domain requires only dropping a new
CSV into that directory — no code changes.

```bash
python -m vernacbridge.dataset_engine
# or, once installed:
vernacbridge-build-dataset
```

This runs the full pipeline — load, validate, merge, deduplicate — and
writes `vernacbridge/datasets/master_dataset.csv` plus
`vernacbridge/datasets/build_report.json`.

## Architecture

```
vernacbridge/
    __init__.py       # VernacBridge facade + module-level convenience functions
    config.py          # Single source of truth: schema, label vocab, thresholds
    exceptions.py       # Typed exception hierarchy
    loader.py           # CSV discovery + raw loading (Dataset Engine)
    validator.py         # Column/label/empty-row validation (Dataset Engine)
    cleaner.py            # Exact + RapidFuzz near-duplicate removal (Dataset Engine)
    dataset_engine.py      # Orchestrates loader -> validator -> cleaner -> master_dataset.csv
    core.py                 # MatchEngine: shared corpus + nearest-neighbor matching
    translator.py             # Translation via MatchEngine
    emotion.py                 # Emotion detection via MatchEngine
    intent.py                   # Intent detection via MatchEngine
    search.py                    # Structured search (by emotion/intent/keyword)
    pipeline.py                   # TranslationPipeline: the structured, extensible orchestrator
    intelligence.py                 # DatasetIntelligence: opt-in match explanations (Phase 2)
    memory.py                        # MemoryEngine + stores: opt-in conversation history (Phase 3)
    context.py                        # ContextEngine: mechanical follow-up resolution (Phase 4)
    protocols.py                       # Structural (typing.Protocol) contracts for every
                                        # injectable collaborator above -- no inheritance required
    datasets/                           # Raw per-domain CSVs + generated master_dataset.csv
                                         # (packaged as real package data -- ships with the
                                         # installed package, not just the source checkout)
tests/                                  # pytest suite (unit + real-dataset integration)
examples/                                # Runnable usage examples
docs/                                     # Per-phase design docs + FUTURE_VISION.md
```

Every runtime engine is built on the same `MatchEngine`, so upgrading the
matching strategy only requires changing `core.py`. Every optional
capability beyond Phase 1 (`context_engine`, `memory_engine`,
`plugin_manager`, `conversation_engine`, `dataset_intelligence`) is
dependency-injected via a `Protocol` in `protocols.py` — any object with
matching methods works, no base class required.

## Running tests

```bash
pytest
```

Unit tests use small, disposable fixture datasets. A separate integration
suite (`tests/test_integration_real_dataset.py`) runs against the real
`vernacbridge/datasets/master_dataset.csv` and is skipped automatically if
that file hasn't been built yet.

## What's next

V1's scope is intentionally Tanglish-only, built on the existing corpus —
see `docs/FUTURE_VISION.md` for ideas that are explicitly deferred (a
professional-communication data domain, multilingual expansion, and other
larger directions) rather than accidentally becoming V1 requirements.
