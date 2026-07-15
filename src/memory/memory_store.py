from __future__ import annotations

import json
import os
import shutil
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
    ) -> None:
        self.config = config or MemoryConfig()
        if memory_dir is not None:
            self.config = self.config.with_memory_dir(memory_dir)
        self.memory_dir = self.config.paths.resolved_memory_dir()
        if not self.memory_dir.is_absolute():
            self.memory_dir = project_root() / self.memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.FILES = {
            "episodic": self.config.paths.episodic_file,
            "failure": self.config.paths.failure_file,
            "semantic_rule": self.config.paths.semantic_file,
        }
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

    @contextmanager
    def _lock(self):
        """Filesystem lock so two processes cannot corrupt the JSONL files."""
        lock_path = self.memory_dir / ".memory.lock"
        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    # ------------------------------------------------------------ atomic I/O

    def _atomic_write_jsonl(self, path: Path, rows: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    def _backup(self, path: Path) -> Path | None:
        """Timestamped backup before rewriting an existing non-empty store."""
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
        with self._lock():
            append_jsonl(self.long_term_path(record.memory_type), record)

    def stage_candidate(self, record: ShortTermMemoryRecord) -> ShortTermMemoryRecord:
        """Upsert one record into short-term memory, keyed by stm_id.

        Re-staging the same candidate (same case rerun) replaces the previous
        row instead of duplicating it."""
        with self._lock():
            existing = self.load_short_term()
            by_id = {row.stm_id: row for row in existing}
            previous = by_id.get(record.stm_id)
            if previous is not None and previous.status in {"promoted", "merged"}:
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
        with self._lock():
            by_id = {row.stm_id: row for row in self.load_short_term()}
            for record in records:
                by_id[record.stm_id] = record
            self._atomic_write_jsonl(self.short_term_path, list(by_id.values()))

    def replace_short_term(self, records: list[ShortTermMemoryRecord]) -> None:
        with self._lock():
            self._backup(self.short_term_path)
            self._atomic_write_jsonl(self.short_term_path, records)

    def upsert_long_term(self, records: list[MemoryRecord]) -> None:
        """Update or insert long-term records, preserving memory IDs, with a
        timestamped backup and an atomic rewrite per affected file."""
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
        with self._lock():
            append_jsonl(self.event_log_path, event)

    def append_usage_event(
        self,
        event: MemoryUsageEvent,
        path_override: Path | None = None,
    ) -> None:
        target = path_override or self.usage_log_path
        append_jsonl(target, event)

    def load_usage_events(self) -> list[MemoryUsageEvent]:
        return [MemoryUsageEvent.model_validate(row) for row in read_jsonl(self.usage_log_path)]

    def load_consolidation_events(self) -> list[ConsolidationEvent]:
        return [ConsolidationEvent.model_validate(row) for row in read_jsonl(self.event_log_path)]

    # -------------------------------------------------------- archive/snapshot

    def archive_records(self, records: list, name: str) -> Path | None:
        """Append replaced/expired/deprecated records to an archive file so they
        are never silently deleted."""
        if not records:
            return None
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        target = self.archive_dir / f"{name}.jsonl"
        for record in records:
            append_jsonl(target, record)
        return target

    def snapshot(self, label: str | None = None) -> Path:
        """Copy the current memory state into a frozen snapshot directory with a
        manifest containing counts and the state hash."""
        stamp = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        target = self.snapshot_dir / stamp
        target.mkdir(parents=True, exist_ok=True)
        copied = []
        filenames = list(self.FILES.values()) + [
            self.config.paths.short_term_file,
            self.config.paths.event_log_file,
        ]
        for filename in filenames:
            source = self.memory_dir / filename
            if source.exists():
                shutil.copy2(source, target / filename)
                copied.append(filename)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_memory_dir": str(self.memory_dir),
            "files": copied,
            "state_hash": self.state_hash(),
            "counts": self.counts(),
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return target

    def state_hash(self, include_short_term: bool = False) -> str:
        """Deterministic hash of the long-term (and optionally short-term) state."""
        parts: list[str] = []
        for record in sorted(self.load_long_term(), key=lambda r: r.memory_id):
            parts.append(json.dumps(record.model_dump(mode="json"), sort_keys=True))
        if include_short_term:
            for record in sorted(self.load_short_term(), key=lambda r: r.stm_id):
                parts.append(json.dumps(record.model_dump(mode="json"), sort_keys=True))
        return stable_hash_text("\n".join(parts), length=64)

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
