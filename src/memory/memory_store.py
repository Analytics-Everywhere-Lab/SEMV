from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.memory.memory_config import MemoryConfig
from src.schemas.memory_schema import (
    ConsolidationEvent,
    MemoryRecord,
    MemoryType,
    MemoryUsageEvent,
    ShortTermMemoryRecord,
)
from src.utils.hashing import stable_hash_text
from src.utils.io import append_jsonl, project_root, read_jsonl, to_jsonable

try:
    import fcntl
except ImportError:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]


class MemoryReadOnlyError(RuntimeError):
    """Raised when a frozen/read-only memory store is asked to mutate state."""


class MemoryStore:
    """JSONL-backed memory store with STM/LTM lifecycles, event logs, archiving,
    snapshots, atomic rewrites, and a filesystem lock."""

    FILES: dict[MemoryType, str] = {
        "episodic": "episodic_memory.jsonl",
        "failure": "failure_memory.jsonl",
        "semantic_rule": "semantic_rules.jsonl",
    }

    def __init__(
        self,
        memory_dir: Path | None = None,
        config: MemoryConfig | None = None,
        read_only: bool | None = None,
    ) -> None:
        requested_dir = Path(memory_dir) if memory_dir is not None else None
        manifest_path = requested_dir / "manifest.json" if requested_dir is not None else None
        if config is None and manifest_path is not None and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = MemoryConfig.model_validate(manifest.get("memory_config") or {})
        self.config = config or MemoryConfig()
        if memory_dir is not None:
            self.config = self.config.with_memory_dir(memory_dir)
        self.read_only = bool(read_only) if read_only is not None else bool(
            requested_dir is not None and (requested_dir / "manifest.json").exists()
        )
        self._lock_state = threading.local()
        self.memory_dir = self.config.paths.resolved_memory_dir()
        if not self.memory_dir.is_absolute():
            self.memory_dir = project_root() / self.memory_dir
        if not self.read_only:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.FILES = {
            "episodic": self.config.paths.episodic_file,
            "failure": self.config.paths.failure_file,
            "semantic_rule": self.config.paths.semantic_file,
        }
        if not self.read_only:
            for filename in self.FILES.values():
                (self.memory_dir / filename).touch(exist_ok=True)
            (self.memory_dir / self.config.paths.short_term_file).touch(exist_ok=True)

    # ------------------------------------------------------------------ paths

    @property
    def short_term_path(self) -> Path:
        return self.memory_dir / self.config.paths.short_term_file

    @property
    def event_log_path(self) -> Path:
        return self.memory_dir / self.config.paths.event_log_file

    @property
    def usage_log_path(self) -> Path:
        return self.memory_dir / self.config.paths.usage_log_file

    @property
    def archive_dir(self) -> Path:
        path = self.config.paths.resolved_archive_dir()
        if not path.is_absolute():
            path = project_root() / path
        return path

    @property
    def snapshot_dir(self) -> Path:
        path = self.config.paths.resolved_snapshot_dir()
        if not path.is_absolute():
            path = project_root() / path
        return path

    def long_term_path(self, memory_type: MemoryType) -> Path:
        return self.memory_dir / self.FILES[memory_type]

    # ------------------------------------------------------------------- lock

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise MemoryReadOnlyError(f"Memory store is read-only: {self.memory_dir}")

    @contextmanager
    def transaction(self):
        """Re-entrant process-safe transaction for a full read/modify/write cycle."""
        if self.read_only:
            yield self
            return
        depth = getattr(self._lock_state, "depth", 0)
        if depth:
            self._lock_state.depth = depth + 1
            try:
                yield self
            finally:
                self._lock_state.depth -= 1
            return
        handle = open(self.memory_dir / ".memory.lock", "a+", encoding="utf-8")
        self._lock_state.depth = 1
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield self
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            self._lock_state.depth = 0

    @contextmanager
    def _lock(self):
        with self.transaction():
            yield

    # ------------------------------------------------------------ atomic I/O

    def _atomic_write_jsonl(self, path: Path, rows: list) -> None:
        self._ensure_writable()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    def _backup(self, path: Path) -> Path | None:
        """Timestamped backup before rewriting an existing non-empty store."""
        self._ensure_writable()
        if not path.exists() or path.stat().st_size == 0:
            return None
        backup_dir = self.archive_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        backup_path = backup_dir / f"{stamp}_{path.name}"
        shutil.copy2(path, backup_path)
        return backup_path

    # ---------------------------------------------------------------- loading

    def load_all(self) -> list[MemoryRecord]:
        """Backward-compatible: load every long-term record regardless of status."""
        return self.load_long_term(statuses=None)

    def load_long_term(
        self,
        memory_types: list[str] | None = None,
        statuses: list[str] | None = None,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for memory_type, filename in self.FILES.items():
            if memory_types is not None and memory_type not in memory_types:
                continue
            for row in read_jsonl(self.memory_dir / filename):
                row.setdefault("memory_type", memory_type)
                record = MemoryRecord.model_validate(row)
                if statuses is not None and record.status not in statuses:
                    continue
                records.append(record)
        return records

    def load_short_term(
        self,
        statuses: list[str] | None = None,
    ) -> list[ShortTermMemoryRecord]:
        records = [
            ShortTermMemoryRecord.model_validate(row)
            for row in read_jsonl(self.short_term_path)
        ]
        if statuses is not None:
            records = [record for record in records if record.status in statuses]
        return records

    # ---------------------------------------------------------------- writing

    def append(self, record: MemoryRecord) -> None:
        """Backward-compatible direct append to a long-term store."""
        self._ensure_writable()
        with self._lock():
            append_jsonl(self.long_term_path(record.memory_type), record)

    def stage_candidate(self, record: ShortTermMemoryRecord) -> ShortTermMemoryRecord:
        """Upsert one record into short-term memory, keyed by stm_id.

        Re-staging the same candidate (same case rerun) replaces the previous
        row instead of duplicating it."""
        self._ensure_writable()
        with self._lock():
            existing = self.load_short_term()
            by_id = {row.stm_id: row for row in existing}
            previous = by_id.get(record.stm_id)
            if previous is not None and (
                previous.status in {"promoted", "merged"}
                or (previous.status == "under_review" and previous.promoted_to_memory_id)
            ):
                # Terminal STM states are preserved: a rerun must not resurrect
                # an already-consolidated observation.
                return previous
            record = record.model_copy(
                update={"updated_at": datetime.now(timezone.utc).isoformat()}
            )
            by_id[record.stm_id] = record
            rows = list(by_id.values())
            overflow = len(rows) - self.config.short_term.max_records
            if overflow > 0:
                rows.sort(key=lambda row: row.staged_at or "")
                dropped, rows = rows[:overflow], rows[overflow:]
                self.archive_records(dropped, "short_term_overflow")
            self._atomic_write_jsonl(self.short_term_path, rows)
        return record

    def upsert_short_term(self, records: list[ShortTermMemoryRecord]) -> None:
        self._ensure_writable()
        with self._lock():
            by_id = {row.stm_id: row for row in self.load_short_term()}
            for record in records:
                by_id[record.stm_id] = record
            self._atomic_write_jsonl(self.short_term_path, list(by_id.values()))

    def replace_short_term(self, records: list[ShortTermMemoryRecord]) -> None:
        self._ensure_writable()
        with self._lock():
            self._backup(self.short_term_path)
            self._atomic_write_jsonl(self.short_term_path, records)

    def upsert_long_term(self, records: list[MemoryRecord]) -> None:
        """Update or insert long-term records, preserving memory IDs, with a
        timestamped backup and an atomic rewrite per affected file."""
        self._ensure_writable()
        if not records:
            return
        with self._lock():
            by_type: dict[str, list[MemoryRecord]] = {}
            for record in records:
                by_type.setdefault(record.memory_type, []).append(record)
            for memory_type, type_records in by_type.items():
                path = self.long_term_path(memory_type)  # type: ignore[arg-type]
                existing_rows = read_jsonl(path)
                by_id: dict[str, dict] = {}
                order: list[str] = []
                for row in existing_rows:
                    row.setdefault("memory_type", memory_type)
                    memory_id = row.get("memory_id", "")
                    if memory_id not in by_id:
                        order.append(memory_id)
                    by_id[memory_id] = row
                for record in type_records:
                    if record.memory_id not in by_id:
                        order.append(record.memory_id)
                    by_id[record.memory_id] = record.model_dump(mode="json")
                self._backup(path)
                self._atomic_write_jsonl(path, [by_id[memory_id] for memory_id in order])

    # ----------------------------------------------------------------- events

    def append_consolidation_event(self, event: ConsolidationEvent) -> None:
        self._ensure_writable()
        with self._lock():
            if event.event_id in {row.event_id for row in self.load_consolidation_events()}:
                return
            append_jsonl(self.event_log_path, event)

    def append_usage_event(
        self,
        event: MemoryUsageEvent,
        path_override: Path | None = None,
    ) -> None:
        if self.read_only and path_override is None:
            self._ensure_writable()
        target = path_override or self.usage_log_path
        if self.read_only and path_override is not None:
            target_abs = target.resolve()
            memory_abs = self.memory_dir.resolve()
            if target_abs == memory_abs or memory_abs in target_abs.parents:
                self._ensure_writable()
        if self.read_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            lock_handle = open(target.with_suffix(target.suffix + ".lock"), "a+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                existing = {row.get("event_id") for row in read_jsonl(target)}
                if event.event_id not in existing:
                    append_jsonl(target, event)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            return
        with self._lock():
            existing = {row.get("event_id") for row in read_jsonl(target)}
            if event.event_id not in existing:
                append_jsonl(target, event)

    def load_usage_events(self) -> list[MemoryUsageEvent]:
        return [MemoryUsageEvent.model_validate(row) for row in read_jsonl(self.usage_log_path)]

    def load_consolidation_events(self) -> list[ConsolidationEvent]:
        return [ConsolidationEvent.model_validate(row) for row in read_jsonl(self.event_log_path)]

    # -------------------------------------------------------- archive/snapshot

    def archive_records(self, records: list, name: str) -> Path | None:
        """Append replaced/expired/deprecated records to an archive file so they
        are never silently deleted."""
        self._ensure_writable()
        if not records:
            return None
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        target = self.archive_dir / f"{name}.jsonl"
        for record in records:
            append_jsonl(target, record)
        return target

    def _state_filenames(self) -> list[str]:
        return list(self.FILES.values()) + [
            self.config.paths.short_term_file,
            self.config.paths.event_log_file,
        ]

    @staticmethod
    def _file_hash(path: Path) -> str:
        content = path.read_bytes() if path.exists() else b""
        return hashlib.sha256(content).hexdigest()

    def _state_config(self) -> dict:
        data = self.config.model_dump(mode="json")
        paths = data.get("paths", {})
        for key in ("memory_dir", "archive_dir", "snapshot_dir", "usage_log_file"):
            paths.pop(key, None)
        return data

    def _state_payload(self) -> dict:
        records = self.load_long_term()
        seed_ids = sorted(record.memory_id for record in records if record.origin == "seed")
        return {
            "schema_version": 1,
            "file_hashes": {
                filename: self._file_hash(self.memory_dir / filename)
                for filename in self._state_filenames()
            },
            "memory_config": self._state_config(),
            "seed": {"version": 1, "memory_ids": seed_ids},
        }

    def snapshot(self, label: str | None = None) -> Path:
        """Take one locked, deterministic full-state snapshot and manifest."""
        self._ensure_writable()
        with self.transaction():
            stamp = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            target = self.snapshot_dir / stamp
            if target.exists():
                raise FileExistsError(f"Snapshot already exists: {target}")
            target.mkdir(parents=True, exist_ok=False)
            for filename in self._state_filenames():
                source = self.memory_dir / filename
                destination = target / filename
                if source.exists():
                    shutil.copy2(source, destination)
                else:
                    destination.write_bytes(b"")
            payload = self._state_payload()
            aggregate = stable_hash_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")), length=64
            )
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_memory_dir": str(self.memory_dir),
                "files": self._state_filenames(),
                **payload,
                "state_hash": aggregate,
                "full_state_hash": aggregate,
                "counts": self.counts(),
            }
            (target / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            return target

    def state_hash(self, include_short_term: bool = True) -> str:
        """Deterministic complete state hash; external usage telemetry is excluded."""
        del include_short_term
        payload = self._state_payload()
        return stable_hash_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), length=64
        )

    def counts(self) -> dict[str, int]:
        long_term = self.load_long_term()
        short_term = self.load_short_term()
        counts: dict[str, int] = {
            "short_term": len(short_term),
            "long_term": len(long_term),
        }
        for record in long_term:
            counts[f"long_term_{record.memory_type}"] = counts.get(f"long_term_{record.memory_type}", 0) + 1
            counts[f"long_term_status_{record.status}"] = counts.get(f"long_term_status_{record.status}", 0) + 1
        for record in short_term:
            counts[f"short_term_status_{record.status}"] = counts.get(f"short_term_status_{record.status}", 0) + 1
        return counts
