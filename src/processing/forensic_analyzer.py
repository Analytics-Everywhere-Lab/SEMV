from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Iterable

from PIL import Image, ImageChops, ImageStat

from src.schemas.case_schema import MediaItem
from src.schemas.evidence_schema import EvidenceItem, Provenance
from src.utils.hashing import stable_hash_text
from src.utils.tool_config import media_config


class ForensicAnalyzer:
    def __init__(self, config: dict | None = None) -> None:
        self.config = media_config(config)

    def analyze(
        self,
        media: MediaItem,
        visual_targets: Iterable[str | Path],
        output_dir: Path,
        base_dir: Path | None = None,
        metadata_items: list[EvidenceItem] | None = None,
    ) -> list[EvidenceItem]:
        if not self.config.get("enable_forensic_adapter", True):
            return [self._uncertainty_item(media.path, "forensic_adapter_disabled")]
        engine = self.config.get("forensic_engine", "basic")

        if engine == "disabled":
            return [self._uncertainty_item(media.path, "forensic_adapter_disabled")]

        if engine not in {"basic", "trufor", "deep"}:
            return [self._uncertainty_item(media.path, f"unknown_forensic_engine:{engine}")]

        output_dir.mkdir(parents=True, exist_ok=True)
        targets = [Path(path) for path in visual_targets]
        if not targets:
            resolved = media.resolved_path(base_dir)
            targets = [resolved] if resolved.exists() else []
        if not targets:
            return [self._uncertainty_item(media.path, "forensic_media_missing")]

        if engine in {"trufor", "deep"}:
            return self._analyze_deep(
                media=media,
                targets=targets,
                output_dir=output_dir,
                base_dir=base_dir,
                metadata_items=metadata_items,
            )

        return self._analyze_basic(
            media=media,
            targets=targets,
            output_dir=output_dir,
            base_dir=base_dir,
            metadata_items=metadata_items,
        )

    def _analyze_basic(
        self,
        media: MediaItem,
        targets: list[Path],
        output_dir: Path,
        base_dir: Path | None,
        metadata_items: list[EvidenceItem] | None,
    ) -> list[EvidenceItem]:
        flags = self._metadata_flags(metadata_items or [])
        ela_paths = []
        noise_scores = []
        blur_scores = []
        border_flags = []
        for target in targets[:8]:
            if not target.exists():
                continue
            try:
                metrics = self._analyze_image(target, output_dir)
            except Exception:  # pragma: no cover - depends on media format
                continue
            ela_paths.extend(metrics.get("ela_paths", []))
            noise_scores.append(metrics.get("noise_score", 0.0))
            blur_scores.append(metrics.get("blur_score", 0.0))
            border_flags.extend(metrics.get("flags", []))
        flags.extend(border_flags)
        if not noise_scores and media.media_type == "image":
            flags.append("forensic_image_unreadable")
        if "metadata_stripped" in flags:
            summary = "Metadata appears stripped; this is common after social media redistribution."
        elif flags:
            summary = "Basic forensic analysis found caution flags: " + ", ".join(sorted(set(flags))) + "."
        else:
            summary = "No strong manipulation cue detected by basic forensic analysis."

        evidence_id = f"forensic_{stable_hash_text(media.path + summary)}"
        return [
            EvidenceItem(
                evidence_id=evidence_id,
                source_type="forensic_analysis",
                source=media.path,
                title="Basic forensic analysis",
                content=summary,
                reliability=0.60,
                relevance=0.80,
                media_path=media.path,
                metadata={
                    "ela_paths": ela_paths,
                    "noise_score": mean(noise_scores) if noise_scores else None,
                    "blur_score": mean(blur_scores) if blur_scores else None,
                    "flags": sorted(set(flags)),
                },
                uncertainty_flags=sorted(set(flags)),
                supports_claim_types=["authenticity"],
                provenance=Provenance(
                    source_id=evidence_id,
                    source_type="forensic_analysis",
                    source=media.path,
                    retrieval_method="basic_forensics",
                ),
            )
        ]

    def _analyze_deep(
        self,
        media: MediaItem,
        targets: list[Path],
        output_dir: Path,
        base_dir: Path | None,
        metadata_items: list[EvidenceItem] | None,
    ) -> list[EvidenceItem]:
        from src.processing.deep_forensics.factory import get_deep_forensic_backend

        max_targets = int(self.config.get("forensic_max_targets", 8))
        candidates = [target for target in targets if target.exists()][:max_targets]

        if not candidates:
            return [self._uncertainty_item(media.path, "deep_forensic_no_valid_targets")]

        try:
            backend = get_deep_forensic_backend(self.config)
            results = backend.analyze_images(candidates, output_dir / "trufor")
        except Exception:
            return self._deep_backend_unavailable(
                media=media,
                candidates=candidates,
                output_dir=output_dir,
                base_dir=base_dir,
                metadata_items=metadata_items,
            )

        all_failed = bool(results) and all(
            result.manipulation_score is None or "deep_forensic_inference_failed" in result.flags
            for result in results
        )
        if all_failed:
            return self._deep_inference_failed(
                media=media,
                candidates=candidates,
                output_dir=output_dir,
                base_dir=base_dir,
                metadata_items=metadata_items,
            )

        scores = [result.manipulation_score for result in results if result.manipulation_score is not None]
        max_score = max(scores) if scores else None
        mean_score = (sum(scores) / len(scores)) if scores else None

        threshold = float(self.config.get("forensic_manipulation_threshold", 0.50))
        high_results = [
            result
            for result in results
            if result.manipulation_score is not None and result.manipulation_score >= threshold
        ]

        flags = sorted({flag for result in results for flag in result.flags})

        if high_results:
            strongest = max(high_results, key=lambda result: result.manipulation_score or 0.0)
            summary = (
                f"Deep forensic analysis using TruFor found manipulation cues in "
                f"{len(high_results)}/{len(results)} visual targets. "
                f"Max score={max_score:.3f}. Strongest target={Path(strongest.target_path).name}. "
                "Treat this as forensic evidence, not final proof."
            )
        elif scores:
            summary = (
                f"Deep forensic analysis using TruFor did not find strong manipulation cues. "
                f"Max score={max_score:.3f}, mean score={mean_score:.3f}."
            )
        else:
            summary = "Deep forensic analysis using TruFor ran, but no valid manipulation score was produced."
            flags = sorted(set(flags) | {"deep_forensic_no_valid_score"})

        evidence_id = f"forensic_deep_{stable_hash_text(media.path + summary)}"

        return [
            EvidenceItem(
                evidence_id=evidence_id,
                source_type="forensic_analysis",
                source=media.path,
                title="Deep forensic analysis",
                content=summary,
                reliability=0.72,
                relevance=0.85,
                media_path=media.path,
                confidence=max_score,
                metadata={
                    "engine": "trufor",
                    "backend": "deep_forensics",
                    "target_count": len(results),
                    "max_manipulation_score": max_score,
                    "mean_manipulation_score": mean_score,
                    "strongest_target": (
                        Path(high_results[0].target_path).name if high_results else None
                    ),
                    "threshold": threshold,
                    "results": [result.__dict__ for result in results],
                    "anomaly_map_paths": [r.anomaly_map_path for r in results if r.anomaly_map_path],
                    "confidence_map_paths": [r.confidence_map_path for r in results if r.confidence_map_path],
                    "heatmap_overlay_paths": [r.heatmap_overlay_path for r in results if r.heatmap_overlay_path],
                },
                uncertainty_flags=flags,
                supports_claim_types=["authenticity"],
                provenance=Provenance(
                    source_id=evidence_id,
                    source_type="forensic_analysis",
                    source=media.path,
                    retrieval_method="deep_forensics_trufor",
                    metadata={"engine": "trufor"},
                ),
            )
        ]

    def _deep_backend_unavailable(
        self,
        media: MediaItem,
        candidates: list[Path],
        output_dir: Path,
        base_dir: Path | None,
        metadata_items: list[EvidenceItem] | None,
    ) -> list[EvidenceItem]:
        return self._deep_fallback_or_uncertainty(
            media=media,
            candidates=candidates,
            output_dir=output_dir,
            base_dir=base_dir,
            metadata_items=metadata_items,
            flag="deep_forensic_backend_unavailable",
        )

    def _deep_inference_failed(
        self,
        media: MediaItem,
        candidates: list[Path],
        output_dir: Path,
        base_dir: Path | None,
        metadata_items: list[EvidenceItem] | None,
    ) -> list[EvidenceItem]:
        return self._deep_fallback_or_uncertainty(
            media=media,
            candidates=candidates,
            output_dir=output_dir,
            base_dir=base_dir,
            metadata_items=metadata_items,
            flag="deep_forensic_inference_failed",
        )

    def _deep_fallback_or_uncertainty(
        self,
        media: MediaItem,
        candidates: list[Path],
        output_dir: Path,
        base_dir: Path | None,
        metadata_items: list[EvidenceItem] | None,
        flag: str,
    ) -> list[EvidenceItem]:
        if self.config.get("forensic_fallback_to_basic", True):
            items = self._analyze_basic(
                media=media,
                targets=candidates,
                output_dir=output_dir / "basic_fallback",
                base_dir=base_dir,
                metadata_items=metadata_items,
            )
            for item in items:
                item.uncertainty_flags = sorted(
                    set(item.uncertainty_flags) | {flag, "deep_forensic_fallback"}
                )
                item.metadata["deep_forensic_fallback"] = True
            return items
        return [self._uncertainty_item(media.path, flag)]

    @staticmethod
    def _metadata_flags(metadata_items: list[EvidenceItem]) -> list[str]:
        flags = []
        for item in metadata_items:
            flags.extend(item.uncertainty_flags)
            software = str(item.metadata).lower()
            if any(token in software for token in ("photoshop", "gimp", "snapseed", "canva")):
                flags.append("software_tag_suspicious")
        return flags

    @staticmethod
    def _analyze_image(path: Path, output_dir: Path) -> dict:
        with Image.open(path) as image:
            image = image.convert("RGB")
            ela_path = output_dir / f"{path.stem}_ela.jpg"
            recompressed = output_dir / f"{path.stem}_ela_tmp.jpg"
            image.save(recompressed, "JPEG", quality=90)
            with Image.open(recompressed) as compressed:
                ela = ImageChops.difference(image, compressed.convert("RGB"))
                extrema = ela.getextrema()
                max_diff = max(channel[1] for channel in extrema) or 1
                scale = 255.0 / max_diff
                ela = ImageChops.multiply(ela, Image.new("RGB", ela.size, (int(scale),) * 3))
                ela.save(ela_path)
            try:
                recompressed.unlink()
            except OSError:
                pass
            stat = ImageStat.Stat(image.convert("L"))
            noise_score = float(stat.stddev[0])
            blur_score = float(stat.var[0])
            flags = []
            if path.suffix.lower() in {".jpg", ".jpeg"} and noise_score < 8:
                flags.append("low_noise_possible_recompression")
            if _has_black_border(image):
                flags.append("black_border_or_recapture_cue")
            return {"ela_paths": [str(ela_path)], "noise_score": noise_score, "blur_score": blur_score, "flags": flags}

    @staticmethod
    def _uncertainty_item(source: str, flag: str) -> EvidenceItem:
        evidence_id = f"uncertainty_{stable_hash_text(source + flag)}"
        return EvidenceItem(
            evidence_id=evidence_id,
            source_type="synthetic_uncertainty",
            source=source,
            title="Forensic analysis unavailable",
            content=f"Forensic analysis was not run for {source} ({flag}).",
            reliability=0.2,
            relevance=0.50,
            media_path=source,
            uncertainty_flags=[flag],
            supports_claim_types=["authenticity"],
            provenance=Provenance(
                source_id=evidence_id,
                source_type="synthetic_uncertainty",
                source=source,
                retrieval_method="local_capability_check",
                metadata={"adapter": "forensics", "flag": flag},
            ),
        )


def _has_black_border(image: Image.Image) -> bool:
    width, height = image.size
    if width < 20 or height < 20:
        return False
    strips = [
        image.crop((0, 0, width, max(1, height // 20))),
        image.crop((0, height - max(1, height // 20), width, height)),
        image.crop((0, 0, max(1, width // 20), height)),
        image.crop((width - max(1, width // 20), 0, width, height)),
    ]
    means = [ImageStat.Stat(strip.convert("L")).mean[0] for strip in strips]
    return any(value < 8 for value in means)
