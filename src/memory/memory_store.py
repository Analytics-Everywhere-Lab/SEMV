from __future__ import annotations

from pathlib import Path

from src.schemas.memory_schema import MemoryRecord, MemoryType
from src.utils.io import append_jsonl, project_root, read_jsonl


class MemoryStore:
    FILES: dict[MemoryType, str] = {
        "episodic": "episodic_memory.jsonl",
        "failure": "failure_memory.jsonl",
        "semantic_rule": "semantic_rules.jsonl",
    }

    def __init__(self, memory_dir: Path | None = None) -> None:
        self.memory_dir = memory_dir or project_root() / "data" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for filename in self.FILES.values():
            (self.memory_dir / filename).touch(exist_ok=True)

    def load_all(self) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for memory_type, filename in self.FILES.items():
            for row in read_jsonl(self.memory_dir / filename):
                row.setdefault("memory_type", memory_type)
                records.append(MemoryRecord.model_validate(row))
        return records

    def append(self, record: MemoryRecord) -> None:
        append_jsonl(self.memory_dir / self.FILES[record.memory_type], record)
