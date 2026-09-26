# VernacBridge — Project Handover

**Status as of this document:** Stable baseline, v0.1.0
**Purpose:** Bring a future session (human or Claude) up to speed without re-deriving decisions already made.

---

## 1. Vision

VernacBridge is a Python NLP library for understanding **Tanglish** (Tamil written in
English letters) — not just a translator. Long-term scope: translation, emotion
detection, intent detection, context understanding, memory, AI integration, and
eventual multilingual expansion, distributed as a pip-installable package (`import
vernacbridge`).

**Standing engineering rules** (established early, still in force):
- Build on the existing architecture — do not redesign without explaining why first.
- Modular Python, clean architecture, production-quality code, no tutorial/demo code.
- Explain architectural decisions before implementing them.

---

## 2. Architecture

Everything downstream of the raw CSVs flows through one shared component,
`MatchEngine`, so upgrading the matching strategy later (e.g. swapping nearest-
neighbor fuzzy matching for embeddings) only requires changing `core.py`.

```
Raw CSVs (datasets/*.csv)
        │
        ▼
┌─────────────────────────── Dataset Engine ───────────────────────────┐
│  DatasetLoader → DatasetValidator → merge → DatasetCleaner            │
│  (discover/read)   (schema + labels)          (exact + near-dup)      │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
  datasets/master_dataset.csv  +  datasets/build_report.json
        │
        ▼
┌────────────────────────────  MatchEngine  ────────────────────────────┐
│  Loads master_dataset.csv into memory once; normalizes + buckets text; │
│  answers "closest known sentence" queries (top_matches / best_match)   │
└─────────────────────────────────────────────────────────────────────┘
        │
        ├──► Translator      (translate)
        ├──► EmotionDetector (detect_emotion)
        ├──► IntentDetector  (detect_intent)
        └──► SearchEngine    (search_by_emotion / intent / keyword)
                │
                ▼
      VernacBridge facade (__init__.py) — one shared MatchEngine
      instance wired into all four runtime engines, plus module-level
      convenience functions (vernacbridge.translate(), etc.)
```

**Design principles baked into the current code:**
- **Single source of truth for schema/labels**: `config.py` holds required columns,
  valid emotion/intent/urgency/confidence vocabularies, and every tunable threshold.
  Adding a domain (e.g. "health") means appending labels here + dropping a CSV in
  `datasets/` — no other code changes.
- **Non-strict validation by default**: invalid emotion/intent labels are dropped
  per-row with a warning, not fatal — datasets are still being actively collected.
- **Structural vs. content validation split**: a missing required *column* fails the
  whole file (nothing usable can be salvaged); a bad *label* or empty field fails
  only that row.
- **Bucketed fuzzy matching**: near-duplicate detection and best-match lookup both
  bucket candidates by a cheap normalized-prefix key (`block_key`) before running
  RapidFuzz, keeping this close to O(n) instead of O(n²) as the corpus grows.
- **Corpus-matching, not generative**: translation/emotion/intent are all nearest-
  neighbor lookups against labelled examples — exact and explainable, appropriate
  for a few-thousand-row curated corpus, but bounded by corpus coverage (see
  Limitations).
- **RapidFuzz with graceful fallback**: `utils.similarity_score()` uses RapidFuzz's
  `token_sort_ratio` when available, falls back to `difflib.SequenceMatcher` with a
  logged warning otherwise. Same API either way.

---

## 3. Folder Structure

```
vernacbridge_project/
├── README.md
├── LICENSE                  (MIT)
├── requirements.txt
├── pyproject.toml           (build config, deps, pytest/ruff/black/mypy config)
├── vernacbridge/
│   ├── __init__.py          # VernacBridge facade + module-level convenience fns
│   ├── config.py             # Schema, label vocab, thresholds — single source of truth
│   ├── exceptions.py          # Typed exception hierarchy (VernacBridgeError base)
│   ├── loader.py               # DatasetLoader — CSV discovery + raw read
│   ├── validator.py             # DatasetValidator — column/label/empty-row checks
│   ├── cleaner.py                 # DatasetCleaner — exact + RapidFuzz near-dup removal
│   ├── dataset_engine.py           # DatasetEngine — orchestrates the pipeline above
│   ├── core.py                      # MatchEngine — shared corpus + nearest-neighbor match
│   ├── translator.py                 # Translator (translate())
│   ├── emotion.py                     # EmotionDetector (detect_emotion())
│   ├── intent.py                       # IntentDetector (detect_intent())
│   ├── search.py                        # SearchEngine (search_by_emotion/intent/keyword)
│   ├── utils.py                          # normalize_text, block_key, similarity_score, logging
│   └── py.typed                           # PEP 561 marker
├── tests/                                  # pytest suite (see §5)
├── datasets/
│   ├── vernacbridge_career_1000.csv
│   ├── vernacbridge_education_1000.csv
│   ├── vernacbridge_family_1000.csv
│   ├── vernacbridge_general_1000.csv
│   ├── master_dataset.csv               # generated by DatasetEngine, not hand-edited
│   └── build_report.json                # generated per-build stats
├── docs/                                  # currently empty — see Roadmap
└── examples/                              # currently empty — see Roadmap
```

Every raw dataset CSV uses the same 7-column schema:
`tanglish_input, english_output, intent, emotion, keywords, urgency, confidence`.

---

## 4. Implemented Features

### Dataset Engine
- Discovers every `*.csv` in `datasets/` (excluding `master_dataset.csv` itself),
  reads with encoding fallback (`utf-8` → `utf-8-sig` → `latin-1`).
- Validates required columns (fatal per-file) and content — empty non-nullable
  fields, invalid emotion/intent labels (dropped per-row, reported, non-fatal).
- Merges all valid rows, then removes exact duplicates and RapidFuzz near-duplicates
  (bucketed, threshold 92.0 by default).
- Writes `master_dataset.csv` and a JSON `build_report.json` with per-file and
  aggregate stats (rows in/out at every stage).
- Runnable via `python -m vernacbridge.dataset_engine` or the `vernacbridge-build-dataset`
  console script.

### Runtime engines (all built on the shared `MatchEngine`)
- **Translator** — `translate(text)` → best-match English translation, confidence
  score, emotion, intent, urgency, matched source sentence. Falls back to returning
  the original text untranslated (confidence-tagged) below the confidence threshold.
- **EmotionDetector** — `detect_emotion(text)` → emotion label + confidence.
- **IntentDetector** — `detect_intent(text)` → intent label + urgency + confidence.
- **SearchEngine** — structured lookups by emotion, intent, or free-text keyword
  (substring match across keywords/Tanglish input/English output).
- **VernacBridge facade** — one shared, lazily-loaded `MatchEngine` wired into all
  four engines; supports a custom `master_dataset_path`. Module-level convenience
  functions (`vernacbridge.translate()`, etc.) back a default singleton instance.

### Verified this session
- Full pipeline run against real, uploaded data: 4,000 rows (career/education/family/
  general, 1,000 each) → 3,775 rows in `master_dataset.csv` (1 exact duplicate, 224
  near-duplicates removed; spot-checked as legitimate template repetition, not
  over-aggressive pruning).
- Existing 87-test pytest suite executed and confirmed: 83 pass outright; the
  remaining 4 fail only under the `difflib` fallback (no RapidFuzz in the verification
  sandbox) and are expected to pass with RapidFuzz installed, which it is in the
  primary dev environment.
- **Two real bugs found and fixed**:
  1. All four runtime engines did `engine or MatchEngine()` in `__init__`. Because
     `MatchEngine` defines `__len__` but not `__bool__`, an unloaded engine evaluated
     as falsy, so any injected engine was silently discarded and replaced with a new
     default one — meaning the `VernacBridge` facade's four sub-engines never
     actually shared one loaded corpus, and custom `master_dataset_path` was
     silently ignored. Fixed to `engine if engine is not None else MatchEngine()`
     in `translator.py`, `emotion.py`, `intent.py`, `search.py`. Verified fix:
     `vb.translator.engine is vb.search_engine.engine` now `True`.
  2. `dataset_engine.py`'s `build()` only caught `DatasetError` around validation,
     but `MissingColumnError` extends the sibling `ValidationError` — so one
     malformed CSV crashed the whole build instead of being skipped into
     `files_failed` as documented. Fixed by catching `(DatasetError, ValidationError)`.
- Packaging gaps filled: `pyproject.toml` referenced `README.md`, `LICENSE`, and
  `vernacbridge/py.typed`, none of which existed — all three added, plus a
  `requirements.txt` matching declared dependencies. A stray leftover directory
  from an earlier shell command was also removed.

---

## 5. Test Suite

`tests/` (pytest, `conftest.py`-driven fixtures):

| File | Covers |
|---|---|
| `test_utils.py` | `normalize_text`, `block_key`, `similarity_score`, RapidFuzz availability flag |
| `test_loader.py` | CSV discovery, encoding fallback, empty/header-only CSVs, partial-failure tolerance |
| `test_validator.py` | Column validation, label validation, empty-field dropping, case normalization |
| `test_cleaner.py` | Exact and near-duplicate removal, threshold respect, pipeline composition |
| `test_dataset_engine.py` | End-to-end build, disk output, missing-column file isolation, all-invalid failure |
| `test_core.py` | `MatchEngine` load/lazy-load, exact/near matching, ordering, `Match.as_dict()` |
| `test_translator_emotion_intent.py` | Confident/low-confidence/empty-input behavior for all three engines, shared-engine state |
| `test_search.py` | Emotion/intent/keyword search, case-insensitivity, limits, empty results |
| `test_integration_real_dataset.py` | Real `master_dataset.csv` via the facade — auto-skips if not yet built |

Run with `pytest` (requires `pip install -e ".[dev]"` or `pip install -r requirements.txt pytest`).

---

## 6. Known Limitations

- **RapidFuzz is a soft dependency with a lower-fidelity fallback.** If unavailable,
  `difflib.SequenceMatcher` is used instead — confirmed to score reordered-word and
  near-duplicate pairs noticeably lower than RapidFuzz's `token_sort_ratio` would.
  This is fine given RapidFuzz is installed in the primary dev environment, but the
  fallback would silently degrade match quality (not correctness) anywhere it isn't.
  Deferred by request; worth hardening later (see Roadmap).
- **Nearest-neighbor, not generative.** Translation/emotion/intent quality is
  entirely bounded by corpus coverage — an input with no sufficiently similar
  labelled example returns `"unknown"` / the original text, no matter how
  linguistically simple it is.
- **Fixed label vocabularies.** `VALID_EMOTIONS` (12 labels) and `VALID_INTENTS`
  (8 labels) in `config.py` are hardcoded. Rows with labels outside these are
  silently dropped (non-strict mode) — expanding coverage requires editing `config.py`.
- **Fuzzy near-duplicate detection is bucketed, not fully O(n).** Bucketing by a
  4-character normalized prefix avoids full O(n²) comparison, but rows that are
  near-duplicates yet start with different prefixes ("enaku..." vs. "naan
  enaku...") would not be compared. Acceptable at current corpus size; documented
  as a scaling consideration.
- **Domain imbalance in near-duplicate removal.** The family dataset lost ~17% of
  rows to near-duplication (vs. ~2–3% for career) due to more repetitive sentence
  templates in that dataset — a data-collection characteristic, not an engine bug,
  but worth knowing when interpreting `build_report.json` deltas across domains.
- **No context or memory layer yet.** Every call is single-turn; there's no
  mechanism for multi-sentence conversation state.
- **`docs/` and `examples/` are empty placeholders.** Structure exists; content doesn't.
- **No CI/CD configured.** Tests run locally only; no GitHub Actions or equivalent yet.
- **Not yet published to PyPI.** `pyproject.toml` is publish-ready in structure, but
  the package has not been built/uploaded.

---

## 7. Recommended Roadmap — v0.2.0

Roughly in priority order:

1. **Harden the RapidFuzz dependency handling.** Options to evaluate: fail loudly
   (raise, not just warn) when RapidFuzz is missing and strict mode is desired; add
   a `require_rapidfuzz` config flag; or improve the `difflib` fallback's token-order
   robustness so degraded-but-not-silent behavior is less of a cliff. (Explicitly
   deferred from this session, first candidate for next.)
2. **Context Engine.** Multi-turn conversation state — likely a new `context.py`
   module consuming `MatchEngine` results across turns rather than replacing it.
   Needs a design discussion before implementation (per standing rule: explain
   before building).
3. **Confidence calibration.** `MATCH_CONFIDENCE_THRESHOLD` (60.0) and
   `NEAR_DUPLICATE_THRESHOLD` (92.0) are currently single global constants. Worth
   evaluating whether they should vary by domain/intent, and whether real precision/
   recall at these thresholds has been measured against held-out data (it hasn't yet).
4. **Expand dataset domains and volume.** Architecture already supports this with
   zero code changes (drop a CSV in `datasets/`) — this is a data-collection task,
   not an engineering one.
5. **`docs/` and `examples/`.** Populate with real usage examples and API reference;
   currently empty despite being referenced in the package structure.
6. **CI pipeline.** Run the pytest suite (and ruff/black/mypy, already configured in
   `pyproject.toml`) automatically on push.
7. **pip packaging and release.** Build and publish to PyPI/TestPyPI once the above
   stabilizes; verify the sdist/wheel actually installs cleanly in a fresh environment
   (not yet tested).
8. **Memory support and AI/LLM integration.** Longer-term vision items — no design
   work done yet; sequence after context understanding is in place.

---

## 8. How to Resume

```bash
# From the project root:
pip install -r requirements.txt        # or: pip install -e ".[dev]"
python -m vernacbridge.dataset_engine  # rebuild master_dataset.csv if datasets/ changed
pytest                                  # confirm baseline still green
```

Anyone (human or Claude) picking this up should read `config.py` first — it's the
single source of truth for schema, labels, and every tunable threshold in the
system — before touching any engine module.
