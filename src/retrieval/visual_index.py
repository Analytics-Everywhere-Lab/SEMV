from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.hashing import sha256_file, stable_hash_text
from src.utils.io import project_root
from src.utils.tool_config import media_config


class VisualIndex:
    def __init__(self, index_dir: str | Path | None = None, config: dict | None = None) -> None:
        cfg = media_config(config)
        self.index_dir = project_root() / (index_dir or cfg.get("visual_index_dir", "data/visual_index"))
        if Path(index_dir or "").is_absolute():
            self.index_dir = Path(index_dir)  # type: ignore[arg-type]
        self.assets_path = self.index_dir / "assets.jsonl"
        self.phash_threshold = int(cfg.get("phash_threshold", 10))
        self.clip_similarity_threshold = float(cfg.get("clip_similarity_threshold", 0.84))

    def add_assets(
        self,
        paths: list[str | Path],
        case_id: str,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        existing = {row.get("sha256") for row in self._rows()}
        with self.assets_path.open("a", encoding="utf-8") as handle:
            for raw_path in paths:
                path = Path(raw_path)
                if not path.exists():
                    continue
                try:
                    digest = sha256_file(path)
                    if digest in existing:
                        continue
                    row = {
                        "asset_id": f"asset_{stable_hash_text(str(path) + digest)}",
                        "path": str(path),
                        "case_id": case_id,
                        "phash": self._phash(path),
                        "sha256": digest,
                        "source_url": source_url,
                        "metadata": metadata or {},
                    }
                    handle.write(json.dumps(row, default=str) + "\n")
                    existing.add(digest)
                except Exception:
                    continue

    def search(self, query_path: str | Path, exclude_case_id: str | None = None) -> list[dict[str, Any]]:
        query = Path(query_path)
        if not query.exists():
            return []
        try:
            query_hash = self._phash(query)
        except Exception:
            return []
        matches = []
        for row in self._rows():
            if exclude_case_id and row.get("case_id") == exclude_case_id:
                continue
            phash = row.get("phash")
            if not phash:
                continue
            distance = self._hash_distance(query_hash, phash)
            if distance <= self.phash_threshold:
                matches.append({**row, "phash_distance": distance, "clip_similarity": None})
        return sorted(matches, key=lambda row: row.get("phash_distance", 999))

    def _rows(self) -> list[dict[str, Any]]:
        if not self.assets_path.exists():
            return []
        rows = []
        for line in self.assets_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    @staticmethod
    def _phash(path: Path) -> str:
        import imagehash

        with Image.open(path) as image:
            return str(imagehash.phash(image))

    @staticmethod
    def _hash_distance(left: str, right: str) -> int:
        import imagehash

        return int(abs(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)))
