"""
Central configuration for VernacBridge.

Every tunable constant lives here so behaviour can be adjusted (or overridden
by downstream applications) without hunting through the codebase. Paths are
resolved relative to the installed package by default but can be overridden
per-instance by the classes that consume them (see ``dataset_engine.py`` and
``core.py``).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

#: Directory that is scanned for raw, per-domain CSV datasets. Lives inside
#: the package itself (not the project root) so it's included as real
#: package data and discoverable the same way whether running from a
#: source checkout, an editable install, or a normal (non-editable) pip
#: install -- all three cases resolve relative to this package's own
#: location, which is always correct regardless of where pip put it.
DEFAULT_DATASET_DIR: Path = PACKAGE_ROOT / "datasets"

#: Location of the single, cleaned dataset produced by the DatasetEngine.
DEFAULT_MASTER_DATASET_PATH: Path = DEFAULT_DATASET_DIR / "master_dataset.csv"

#: Location of the JSON report written after every dataset build.
DEFAULT_BUILD_REPORT_PATH: Path = DEFAULT_DATASET_DIR / "build_report.json"

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
#: Columns every raw dataset CSV must contain.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "tanglish_input",
    "english_output",
    "intent",
    "emotion",
    "keywords",
    "urgency",
    "confidence",
)

#: Columns that must not be blank/NaN on any retained row.
NON_NULLABLE_COLUMNS: tuple[str, ...] = (
    "tanglish_input",
    "english_output",
    "intent",
    "emotion",
)

# --------------------------------------------------------------------------
# Label vocabularies
# --------------------------------------------------------------------------
# These vocabularies are intentionally kept in one place so that adding a
# new domain (e.g. "health", "finance") only requires appending an intent
# here plus dropping a new CSV into the datasets folder -- no other code
# needs to change.
VALID_EMOTIONS: frozenset[str] = frozenset(
    {
        "angry",
        "anxious",
        "calm",
        "confused",
        "excited",
        "fear",
        "guilty",
        "happy",
        "hopeful",
        "lonely",
        "sad",
        "stressed",
    }
)

VALID_INTENTS: frozenset[str] = frozenset(
    {
        "career",
        "daily_conversation",
        "education",
        "emergency",
        "family",
        "mental_health",
        "motivation",
        "relationship",
    }
)

VALID_URGENCY: frozenset[str] = frozenset({"low", "medium", "high"})

VALID_CONFIDENCE: frozenset[str] = frozenset({"low", "medium", "high"})

# --------------------------------------------------------------------------
# Cleaning / matching thresholds
# --------------------------------------------------------------------------
#: RapidFuzz similarity (0-100) above which two rows are considered
#: near-duplicates during dataset cleaning.
NEAR_DUPLICATE_THRESHOLD: float = 92.0

#: RapidFuzz similarity (0-100) below which a translate/detect call is
#: considered "no confident match" and a fallback response is returned.
MATCH_CONFIDENCE_THRESHOLD: float = 60.0

#: Number of closest candidates returned by TranslationPipeline when the
#: best match falls below MATCH_CONFIDENCE_THRESHOLD.
PIPELINE_TOP_N: int = 3

#: Score gap (0-100) between the winning candidate and its runner-up below
#: which DatasetIntelligence flags a match as a "close call" -- used only
#: by explain-mode output (Phase 2); has no effect on matching itself.
CLOSE_CALL_MARGIN: float = 10.0

#: Maximum number of turns MemoryEngine's default InMemoryStore retains per
#: session before evicting the oldest (Phase 3). Only bounds short-term
#: (in-RAM) history; a JSONFileStore, if configured, retains everything.
MEMORY_SHORT_TERM_CAPACITY: int = 50

#: Word count at or below which ContextEngine treats input as a possible
#: follow-up needing prior-turn context (e.g. "Why?") rather than a
#: complete, standalone sentence (Phase 4). A structural signal, not a
#: language-specific one -- deliberately not a hardcoded question-word
#: list, to avoid guessing at Tanglish vocabulary.
CONTEXT_FOLLOWUP_MAX_WORDS: int = 3

#: Number of most-recent prior turns ContextEngine considers when
#: resolving a follow-up (Phase 4). Starts at 1 (immediately preceding
#: turn only) -- the simplest case the original spec's example needs.
CONTEXT_RECENCY_WINDOW: int = 1


#: Maximum number of most-recent turns MemoryEngine's default short-term
#: store retains per session before evicting the oldest (Phase 3). Has no
#: effect on long-term (persisted) storage, which is uncapped.
MEMORY_SHORT_TERM_CAPACITY: int = 50

#: Number of leading normalized characters used to "bucket" rows before
#: near-duplicate comparison, keeping cleaning close to O(n) instead of
#: O(n^2) on large corpora.
DEDUP_BLOCK_KEY_LENGTH: int = 4

# --------------------------------------------------------------------------
# Encodings attempted (in order) when reading a CSV file.
# --------------------------------------------------------------------------
CANDIDATE_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
