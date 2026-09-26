"""
Custom exception hierarchy for VernacBridge.

All exceptions raised by the library inherit from :class:`VernacBridgeError`
so that calling code can catch every library-specific failure with a single
``except VernacBridgeError`` clause, while still being able to handle more
specific failure modes individually.
"""

from __future__ import annotations


class VernacBridgeError(Exception):
    """Base class for every exception raised by VernacBridge."""


# --------------------------------------------------------------------------
# Dataset discovery / I/O errors
# --------------------------------------------------------------------------
class DatasetError(VernacBridgeError):
    """Base class for dataset related errors."""


class DatasetDirectoryNotFoundError(DatasetError):
    """Raised when the configured datasets directory does not exist."""


class NoDatasetsFoundError(DatasetError):
    """Raised when the datasets directory contains no usable CSV files."""


class DatasetReadError(DatasetError):
    """Raised when a CSV file cannot be parsed (bad encoding, malformed, ...)."""


class EmptyDatasetError(DatasetError):
    """Raised when a CSV file has no data rows after loading."""


# --------------------------------------------------------------------------
# Validation errors
# --------------------------------------------------------------------------
class ValidationError(VernacBridgeError):
    """Base class for validation related errors."""


class MissingColumnError(ValidationError):
    """Raised when a required column is missing from a dataset."""


class InvalidEmotionLabelError(ValidationError):
    """Raised when a row references an emotion label outside the known set."""


class InvalidIntentLabelError(ValidationError):
    """Raised when a row references an intent label outside the known set."""


# --------------------------------------------------------------------------
# Runtime / matching errors
# --------------------------------------------------------------------------
class MasterDatasetNotBuiltError(VernacBridgeError):
    """Raised when a matching engine is used before a master dataset exists."""


class NoMatchFoundError(VernacBridgeError):
    """Raised when no sufficiently similar sentence exists in the corpus."""
