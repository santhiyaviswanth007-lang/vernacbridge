"""
Shared, dependency-light helpers used across VernacBridge modules.

This module intentionally has no VernacBridge-internal imports (other than
``config``) so it can be imported early and safely from anywhere in the
package without risking circular imports.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Final

from . import config

_LOGGING_CONFIGURED = False

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger configured with VernacBridge's standard
    format. Safe to call repeatedly; the root handler is only attached once.
    """
    global _LOGGING_CONFIGURED
    logger = logging.getLogger(name)

    if not _LOGGING_CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
        )
        root = logging.getLogger("vernacbridge")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _LOGGING_CONFIGURED = True

    return logger


logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s]", re.UNICODE)


@lru_cache(maxsize=8192)
def normalize_text(text: str) -> str:
    """
    Normalize Tanglish text for matching purposes only.

    This is deliberately conservative: it lowercases, strips accents,
    removes punctuation and collapses whitespace. It must never be used to
    alter text that will be shown to a user (e.g. ``english_output``) since
    it is lossy by design.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def block_key(text: str, length: int = config.DEDUP_BLOCK_KEY_LENGTH) -> str:
    """
    Cheap bucketing key used to avoid full O(n^2) comparisons during
    near-duplicate detection and matching. Rows/queries only need to be
    compared against others sharing the same block key.
    """
    normalized = normalize_text(text)
    return normalized[:length]


# --------------------------------------------------------------------------
# Similarity scoring (RapidFuzz with a graceful fallback)
# --------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore

    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when rapidfuzz is missing
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning(
        "rapidfuzz is not installed; falling back to difflib for similarity "
        "scoring. Install rapidfuzz for accuracy and performance: "
        "pip install rapidfuzz"
    )


def similarity_score(a: str, b: str) -> float:
    """
    Return a similarity score in the range [0, 100] between two strings.

    Uses RapidFuzz's ``token_sort_ratio`` (robust to word order and length
    differences, well suited to short conversational sentences) when
    available, otherwise falls back to :mod:`difflib`.
    """
    if not a or not b:
        return 0.0

    if _RAPIDFUZZ_AVAILABLE:
        return float(_rf_fuzz.token_sort_ratio(a, b))

    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio() * 100.0


def is_rapidfuzz_available() -> bool:
    """Expose RapidFuzz availability, mainly for diagnostics and tests."""
    return _RAPIDFUZZ_AVAILABLE
