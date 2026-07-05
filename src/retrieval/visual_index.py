from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.hashing import sha256_file, stable_hash_text
from src.utils.io import project_root
from src.utils.tool_config import media_config


class VisualIndex:
    """Persistent local visual index with pHash and optional CLIP/FAISS search."""

    def __init__(self, index_dir: str | Path | None = None, config: dict | None = None) -> None:
        cfg = media_config(config)
        self.config = cfg
        self.methods = set(cfg.get("local_reverse_methods") or ["phash"])
        self.index_dir = project_root() / (index_dir or cfg.get("visual_index_dir", "data/visual_index"))
        if index_dir is not None and Path(index_dir).is_absolute():
            self.index_dir = Path(index_dir)
        self.assets_path = self.index_dir / "assets.jsonl"
        self.phash_path = self.index_dir / "phash.jsonl"
        self.clip_index_path = self.index_dir / "clip.index.faiss"
        self.clip_meta_path = self.index_dir / "clip_meta.jsonl"
        self.phash_threshold = int(cfg.get("phash_threshold", 10))
        self.clip_similarity_threshold = float(cfg.get("clip_similarity_threshold", 0.84))
        self.clip_model_name = str(cfg.get("clip_model_name", "ViT-B-32"))
        self.clip_pretrained = str(cfg.get("clip_pretrained", "openai"))
        self._clip_backend: tuple[Any, Any, Any, Any] | None = None

    def add_assets(
        self,
        paths: list[str | Path],
        case_id: str,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        rows = self._rows()
        existing_sha = {row.get("sha256") for row in rows}
        next_vector_id = self._next_clip_vector_id()
        assets_to_write = []
        phashes_to_write = []
        clip_rows = []
        clip_vectors = []

        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            try:
                digest = sha256_file(path)
                if digest in existing_sha:
                    continue
                phash = self._phash(path) if "phash" in self.methods else None
                asset_id = f"asset_{stable_hash_text(str(path) + digest)}"
                clip_vector_id = None
                if "clip_faiss" in self.methods:
                    vector = self._clip_embedding(path)
                    if vector is not None:
                        clip_vector_id = next_vector_id
                        next_vector_id += 1
                        clip_vectors.append(vector)
                        clip_rows.append({"clip_vector_id": clip_vector_id, "asset_id": asset_id})
                row = {
                    "asset_id": asset_id,
                    "case_id": case_id,
                    "path": str(path),
                    "sha256": digest,
                    "phash": phash,
                    "clip_vector_id": clip_vector_id,
                    "source_url": source_url,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata or {},
                }
                assets_to_write.append(row)
                if phash:
                    phashes_to_write.append({"asset_id": asset_id, "phash": phash})
                existing_sha.add(digest)
            except Exception:
                continue

        self._append_jsonl(self.assets_path, assets_to_write)
        self._append_jsonl(self.phash_path, phashes_to_write)
        if clip_vectors:
            self._append_clip_vectors(clip_vectors)
            self._append_jsonl(self.clip_meta_path, clip_rows)

    def search(self, query_path: str | Path, exclude_case_id: str | None = None) -> list[dict[str, Any]]:
        query = Path(query_path)
        if not query.exists():
            return []
        query_sha = None
        try:
            query_sha = sha256_file(query)
        except Exception:
            pass

        matches: dict[str, dict[str, Any]] = {}
        if "phash" in self.methods:
            for match in self._search_phash(query, exclude_case_id, query_sha):
                _merge_match(matches, match, "phash")
        if "clip_faiss" in self.methods:
            for match in self._search_clip(query, exclude_case_id, query_sha):
                _merge_match(matches, match, "clip_faiss")
        return sorted(matches.values(), key=_match_sort_key)

    def compare_paths(self, query_path: str | Path, candidate_path: str | Path) -> dict[str, Any] | None:
        query = Path(query_path)
        candidate = Path(candidate_path)
        if not query.exists() or not candidate.exists():
            return None
        result: dict[str, Any] = {"methods": []}
        if "phash" in self.methods:
            try:
                distance = self._hash_distance(self._phash(query), self._phash(candidate))
                if distance <= self.phash_threshold:
                    result["phash_distance"] = distance
                    result["methods"].append("phash")
            except Exception:
                pass
        if "clip_faiss" in self.methods:
            qvec = self._clip_embedding(query)
            cvec = self._clip_embedding(candidate)
            if qvec is not None and cvec is not None:
                np = self._numpy_module()
                if np is None:
                    return result if result["methods"] else None
                similarity = float(np.dot(qvec, cvec))
                if similarity >= self.clip_similarity_threshold:
                    result["clip_similarity"] = similarity
                    result["methods"].append("clip_faiss")
        return result if result["methods"] else None

    def _search_phash(
        self,
        query: Path,
        exclude_case_id: str | None,
        query_sha: str | None,
    ) -> list[dict[str, Any]]:
        try:
            query_hash = self._phash(query)
        except Exception:
            return []
        matches = []
        for row in self._rows():
            if self._should_skip(row, query, query_sha, exclude_case_id):
                continue
            phash = row.get("phash")
            if not phash:
                continue
            try:
                distance = self._hash_distance(query_hash, phash)
            except Exception:
                continue
            if distance <= self.phash_threshold:
                matches.append({**row, "phash_distance": distance, "clip_similarity": None, "methods": ["phash"]})
        return matches

    def _search_clip(
        self,
        query: Path,
        exclude_case_id: str | None,
        query_sha: str | None,
    ) -> list[dict[str, Any]]:
        query_vec = self._clip_embedding(query)
        if query_vec is None:
            return []
        faiss = self._faiss_module()
        np = self._numpy_module()
        if faiss is None or np is None or not self.clip_index_path.exists():
            return []
        try:
            index = faiss.read_index(str(self.clip_index_path))
            if index.ntotal == 0:
                return []
            scores, ids = index.search(np.asarray([query_vec], dtype="float32"), min(10, index.ntotal))
        except Exception:
            return []
        meta_by_vector = {int(row["clip_vector_id"]): row for row in self._jsonl_rows(self.clip_meta_path) if row.get("clip_vector_id") is not None}
        assets = {row.get("asset_id"): row for row in self._rows()}
        matches = []
        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0 or float(score) < self.clip_similarity_threshold:
                continue
            meta = meta_by_vector.get(int(vector_id))
            row = assets.get(meta.get("asset_id") if meta else None)
            if not row or self._should_skip(row, query, query_sha, exclude_case_id):
                continue
            matches.append({**row, "phash_distance": None, "clip_similarity": float(score), "methods": ["clip_faiss"]})
        return matches

    def _should_skip(self, row: dict[str, Any], query: Path, query_sha: str | None, exclude_case_id: str | None) -> bool:
        row_path = Path(str(row.get("path", "")))
        if row_path == query:
            return True
        if exclude_case_id and row.get("case_id") == exclude_case_id and row.get("sha256") == query_sha:
            return True
        return False

    def _rows(self) -> list[dict[str, Any]]:
        return self._jsonl_rows(self.assets_path)

    @staticmethod
    def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    @staticmethod
    def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")

    def _append_clip_vectors(self, vectors: list[Any]) -> None:
        faiss = self._faiss_module()
        np = self._numpy_module()
        if faiss is None or np is None:
            return
        matrix = np.asarray(vectors, dtype="float32")
        if self.clip_index_path.exists():
            index = faiss.read_index(str(self.clip_index_path))
        else:
            index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(self.clip_index_path))

    def _next_clip_vector_id(self) -> int:
        rows = self._jsonl_rows(self.clip_meta_path)
        if not rows:
            return 0
        return max(int(row.get("clip_vector_id", -1)) for row in rows) + 1

    def _clip_embedding(self, path: Path) -> Any | None:
        backend = self._load_clip_backend()
        if backend is None:
            return None
        open_clip, torch, model, preprocess = backend
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            device = "cuda" if getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
            model = model.to(device)
            image = image.to(device)
            with torch.no_grad():
                vector = model.encode_image(image)
                vector = vector / vector.norm(dim=-1, keepdim=True)
            return vector.detach().cpu().numpy().astype("float32")[0]
        except Exception:
            return None

    def _load_clip_backend(self) -> tuple[Any, Any, Any, Any] | None:
        if self._clip_backend is not None:
            return self._clip_backend
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.clip_pretrained,
            )
            model.eval()
            self._clip_backend = (open_clip, torch, model, preprocess)
            return self._clip_backend
        except Exception:
            return None

    @staticmethod
    def _faiss_module() -> Any | None:
        try:
            import faiss

            return faiss
        except Exception:
            return None

    @staticmethod
    def _numpy_module() -> Any | None:
        try:
            import numpy

            return numpy
        except Exception:
            return None

    @staticmethod
    def _phash(path: Path) -> str:
        try:
            import imagehash

            with Image.open(path) as image:
                return str(imagehash.phash(image))
        except ImportError:
            return VisualIndex._fallback_phash(path)

    @staticmethod
    def _hash_distance(left: str, right: str) -> int:
        try:
            import imagehash

            return int(abs(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)))
        except ImportError:
            return (int(left, 16) ^ int(right, 16)).bit_count()

    @staticmethod
    def _fallback_phash(path: Path) -> str:
        size = 32
        with Image.open(path) as image:
            resized = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            pixels = [float(value) for value in resized.getdata()]

        coeffs: list[float] = []
        for vertical in range(8):
            for horizontal in range(8):
                scale = VisualIndex._dct_scale(horizontal, size) * VisualIndex._dct_scale(vertical, size)
                total = 0.0
                for y in range(size):
                    y_basis = math.cos(math.pi * (2 * y + 1) * vertical / (2 * size))
                    row_offset = y * size
                    for x in range(size):
                        x_basis = math.cos(math.pi * (2 * x + 1) * horizontal / (2 * size))
                        total += pixels[row_offset + x] * x_basis * y_basis
                coeffs.append(scale * total)

        median = sorted(coeffs[1:])[len(coeffs[1:]) // 2]
        value = 0
        for coeff in coeffs:
            value = (value << 1) | int(coeff > median)
        return f"{value:016x}"

    @staticmethod
    def _dct_scale(index: int, size: int) -> float:
        return math.sqrt(1 / size) if index == 0 else math.sqrt(2 / size)


def _merge_match(matches: dict[str, dict[str, Any]], match: dict[str, Any], method: str) -> None:
    asset_id = str(match.get("asset_id") or match.get("path"))
    existing = matches.get(asset_id)
    if not existing:
        match["methods"] = sorted(set(match.get("methods", []) + [method]))
        matches[asset_id] = match
        return
    existing["methods"] = sorted(set(existing.get("methods", []) + match.get("methods", []) + [method]))
    if match.get("phash_distance") is not None:
        old = existing.get("phash_distance")
        existing["phash_distance"] = match["phash_distance"] if old is None else min(old, match["phash_distance"])
    if match.get("clip_similarity") is not None:
        old = existing.get("clip_similarity")
        existing["clip_similarity"] = match["clip_similarity"] if old is None else max(old, match["clip_similarity"])


def _match_sort_key(match: dict[str, Any]) -> tuple[float, float]:
    phash_score = 1.0 - min(float(match.get("phash_distance") or 64), 64.0) / 64.0
    clip_score = float(match.get("clip_similarity") or 0.0)
    return (-max(phash_score, clip_score), float(match.get("phash_distance") or 999))
