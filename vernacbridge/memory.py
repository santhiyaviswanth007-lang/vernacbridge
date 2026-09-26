"""
Memory Layer (Phase 3).

Implements vernacbridge.protocols.MemoryEngineProtocol, which Phase 1
already defined and already wired into TranslationPipeline._finish()
(``self.memory_engine.record(session_id, result)``, non-fatal on failure).
No change to pipeline.py was needed for this phase -- the write path
already existed and simply had nothing plugged into it until now.

Design
------
"Short-term", "long-term", and "session" memory (as named in the original
spec) are modeled as two swappable stores plus one correlation key, not
three separate subsystems:

- short-term: an in-RAM, per-session, capped history (default:
  InMemoryStore, capacity config.MEMORY_SHORT_TERM_CAPACITY). Always
  active. Gone on process restart.
- long-term: an optional, disk-persisted, uncapped history (JSONFileStore).
  Opt-in -- MemoryEngine does not create one unless you inject it.
- session: not a physical store at all. ``session_id`` is the correlation
  key already threaded through Phase 1's entire pipeline; "session memory"
  is simply ``recall(session_id)`` scoped to that key against whichever
  store(s) are active.

MemoryEngine itself defaults to *nothing persisted beyond RAM* -- injecting
a MemoryEngine at all is already an opt-in on VernacBridge's part (it stays
None by default, exactly like context_engine/plugin_manager/
conversation_engine), and injecting a JSONFileStore as long_term is a
second, separate opt-in for anyone who additionally wants persistence
across restarts. Two deliberate gates, not one -- see docs/
PHASE_3_MEMORY_LAYER.md for the reasoning.

Nothing in TranslationPipeline calls recall() yet. Reading memory back to
resolve context ("Why?" -> "Why am I afraid?") is Context Engine's job
(Phase 4, not yet built). MemoryEngine is independently queryable today
(call .recall() directly, or use it in a test), but the pipeline itself
only ever writes to it in this phase.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from . import config
from .utils import get_logger

if TYPE_CHECKING:
    from .pipeline import PipelineResult

logger = get_logger(__name__)


class MemoryStore(Protocol):
    """
    Internal storage contract MemoryEngine's stores must satisfy.

    Deliberately kept here rather than in protocols.py: this is
    MemoryEngine's own swappable collaborator, not something
    TranslationPipeline injects directly -- protocols.py stays scoped to
    contracts TranslationPipeline itself depends on.
    """

    def append(self, session_id: str | None, result: "PipelineResult") -> None:
        """Persist one turn for ``session_id``."""
        ...

    def get(self, session_id: str | None) -> list["PipelineResult"]:
        """Return all turns persisted for ``session_id``, oldest first."""
        ...


class InMemoryStore:
    """
    Default short-term store: per-session, capped, RAM-only.

    Backed by ``collections.deque(maxlen=capacity)``, so the oldest turn is
    evicted automatically once a session's history exceeds ``capacity`` --
    no manual eviction logic needed.
    """

    def __init__(self, capacity: int = config.MEMORY_SHORT_TERM_CAPACITY) -> None:
        self.capacity = capacity
        self._data: dict[str | None, deque] = defaultdict(lambda: deque(maxlen=capacity))

    def append(self, session_id: str | None, result: "PipelineResult") -> None:
        self._data[session_id].append(result)

    def get(self, session_id: str | None) -> list["PipelineResult"]:
        return list(self._data.get(session_id, []))


class JSONFileStore:
    """
    Optional long-term store: append-only JSON-lines file, uncapped,
    survives process restarts.

    Each line is ``{"session_id": ..., "result": {...PipelineResult...}}``.
    Not a database -- deliberately simple, matching the project's existing
    philosophy (established in Phase 2) of producing durable data without
    reaching for infrastructure the project doesn't need yet. Reads scan
    the whole file, which is fine at conversation-log scale; if this ever
    becomes a bottleneck, swapping in a different MemoryStore
    implementation requires no changes to MemoryEngine or pipeline.py.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, session_id: str | None, result: "PipelineResult") -> None:
        record = {"session_id": session_id, "result": result}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get(self, session_id: str | None) -> list["PipelineResult"]:
        if not self.path.exists():
            return []
        results: list["PipelineResult"] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["session_id"] == session_id:
                    results.append(record["result"])
        return results


class MemoryEngine:
    """
    Default implementation of :class:`vernacbridge.protocols.
    MemoryEngineProtocol`.

    ``short_term`` defaults to a fresh :class:`InMemoryStore`. ``long_term``
    defaults to ``None`` -- persistence across restarts is opt-in, injected
    explicitly (e.g. ``MemoryEngine(long_term=JSONFileStore("history.jsonl"))``).

    ``record()`` always writes to short-term, and additionally to
    long-term if configured. ``recall()`` prefers long-term when present
    (it's the uncapped, authoritative history); otherwise it falls back to
    short-term.
    """

    def __init__(
        self,
        short_term: MemoryStore | None = None,
        long_term: MemoryStore | None = None,
    ) -> None:
        self.short_term: MemoryStore = short_term if short_term is not None else InMemoryStore()
        self.long_term: MemoryStore | None = long_term

    def record(self, session_id: str | None, result: "PipelineResult") -> None:
        self.short_term.append(session_id, result)
        if self.long_term is not None:
            self.long_term.append(session_id, result)

    def recall(self, session_id: str | None) -> list["PipelineResult"]:
        if self.long_term is not None:
            return self.long_term.get(session_id)
        return self.short_term.get(session_id)
