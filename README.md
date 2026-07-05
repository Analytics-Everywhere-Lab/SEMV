# Self-Evolving Contestable QBAF for Multimedia Verification

Reference implementation for a self-evolving, contestable Quantitative Bipolar
Argumentation Framework (QBAF) for multimedia verification. The system verifies
image, video, and image-caption claims by decomposing them into scoped
subclaims, collecting multimodal evidence, constructing support and attack
arguments, propagating argument strength over QBAF graphs, and producing both
machine-readable and human-readable verification reports.

The project is designed for research on multimedia verification pipelines that
must remain auditable after prediction. In addition to standard inference and
evaluation, the repository includes memory-enabled self-evolution hooks,
uncertainty tracking, and a human-contestation interface for reviewing and
correcting model-generated arguments.

This repository is organized as research code for running single-case inference,
canonical dataset conversion, memory-enabled ablations, and MV2026/COSMOS-style
evaluation protocols.

## Method Overview

For each case, the pipeline performs the following stages:

1. Canonicalize the input as a `CaseBundle`.
2. Load media assets and extract real metadata, scene-aware video keyframes,
   OCR, ASR, VLM observations, forensic signals, and local reverse-image matches.
3. Use provided claims or decompose the main claim into scoped verification
   subclaims.
4. Retrieve relevant verified memory when enabled by the case run configuration.
5. Plan research and optionally run cached, free-web, geolocation, and
   reverse-search evidence retrieval while reusing existing media evidence.
6. Normalize evidence and construct an evidence graph.
7. Generate, verify, and score support/attack arguments per subclaim.
8. Build and propagate QBAF graphs, resolving clashes when required.
9. Aggregate subclaim decisions into a final verification label and confidence.
10. Render structured JSON and Markdown reports with evidence, media analysis,
    argument, QBAF, uncertainty, and memory traces.
11. Optionally reflect after prediction to produce verified memory-update
    candidates.

Gold labels and gold reports are guarded by leakage checks and are only used
post-prediction in self-evolving or bootstrap-memory modes.

## Human Contestation and Adaptive Revision

SEMV is intended to support contestable verification rather than a closed
one-shot prediction. The human reviewer does not need to directly edit every
internal pipeline artifact. Instead, the contestation interface is centered on
arguments, because arguments are the bridge between evidence, subclaims, QBAF
reasoning, and the final decision.

The recommended reviewer action space is deliberately small:

- `accept`: confirm that an existing argument is valid and should remain active.
- `reject`: mark an existing argument as invalid, unsupported, misleading, or
  irrelevant.
- `edit`: correct the text, stance, score hint, linked evidence, or scope of an
  existing argument.
- `add`: introduce a missing support or attack argument for a subclaim.

A reviewer should normally inspect all generated arguments in one batch. This
keeps the human workflow simple while still allowing the framework to infer
where revision is needed. Each action can include a `revision_target` metadata
field telling the system which stage should be revisited.

Suggested `revision_target` values are:

```text
claim_decomposition
media_processing
retrieval
evidence_normalization
argument_generation
argument_verification
argument_scoring
qbaf_propagation
final_aggregation
reporting
```

This makes the contestation mechanism adaptive. For example, rejecting an
argument because the retrieved source does not actually support the claim should
send the pipeline back to `retrieval`, not merely rescore the argument. Editing
an argument's stance should restart from `argument_verification` or
`argument_scoring`. Adding a missing counterargument can restart from
`qbaf_propagation` if the evidence is already present, or from `retrieval` if new
evidence is required.

A typical feedback file can use the following shape:

```json
{
  "case_id": "ID333",
  "reviewer_id": "human_1",
  "review_scope": "all_arguments",
  "actions": [
    {
      "action": "reject",
      "argument_id": "arg_where_002",
      "reason": "The cited source describes a similar scene but not the claimed location.",
      "revision_target": "retrieval"
    },
    {
      "action": "edit",
      "argument_id": "arg_when_001",
      "revised_text": "The evidence supports that the video was online by the publication date, but not the exact recording date.",
      "stance": "attack",
      "score_hint": 0.72,
      "revision_target": "argument_scoring"
    },
    {
      "action": "add",
      "claim_id": "authenticity_1",
      "stance": "support",
      "text": "No visible editing artifacts are reported by the available forensic evidence.",
      "evidence_ids": ["ev_forensic_001"],
      "revision_target": "qbaf_propagation"
    }
  ]
}
```

The current codebase already contains the main contestability hooks:

- `CaseBundle.run_config.allow_human_contestation` controls whether a case is
  intended to allow human review.
- `scripts/run_case.py` exposes `--human-feedback-json` as the CLI entry point
  for feedback files.
- `VerificationReport.reflection_logs[*].human_feedback` provides a structured
  place to preserve human feedback during reflection.
- `report.md` includes a contestation-log section.

Important implementation note: in the current ZIP, the canonical
`run_case_bundle` path does not yet fully consume `--human-feedback-json` to
perform automatic targeted re-execution. The README therefore treats the JSON
above as the intended integration contract for the contestation/revision module.
A complete implementation should parse the feedback file, map each action to the
lowest affected stage, re-run downstream stages only, and persist a contested
report alongside the original report.

## Repository Layout

```text
configs/                  Runtime, scoring, memory, tool, and evaluation configs
data/cases/               Small local example cases
data/memory/              Episodic, semantic, and failure-memory stores
data/evidence_cache/      Cached/sample evidence records
scripts/                  CLI entry points for runs, conversion, and evaluation
src/aggregation/          Final decision aggregation
src/argumentation/        Argument generation, verification, scoring, clash handling
src/evaluation/           MV2026/COSMOS metrics and protocol runner
src/evidence/             Evidence normalization, provenance, and graph building
src/ingestion/            Dataset adapters and canonical bundle writer
src/memory/               Memory retrieval, verification, consolidation, seeding
src/planning/             Claim decomposition and research planning
src/processing/           Media loading, metadata, OCR, ASR, VLM, forensics, keyframes
src/qbaf/                 QBAF graph construction, propagation, decision mapping
src/reflection/           Failure analysis and memory-update candidate generation
src/reporting/            JSON and Markdown report rendering
tests/                    Unit and integration tests
```

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies include `pydantic`, `PyYAML`, `requests`, `Pillow`, and
`pytest`. Real media processing also uses `opencv-python-headless`, `numpy`,
`scenedetect`, `easyocr`, `faster-whisper`, `imagehash`, `open-clip-torch`,
`faiss-cpu`, `beautifulsoup4`, `trafilatura`, and `duckduckgo_search`. Some
adapters also require external local binaries or services, noted below.

## LLM Backend

All agent-like components share `OllamaLLMClient`. Configure the local Ollama
endpoint and model in `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=your_local_model_name
OLLAMA_TEMPERATURE=0.0
OLLAMA_NUM_CTX=8192
OLLAMA_TIMEOUT=120
```

`OLLAMA_MODEL` must name a model already available to the local Ollama server.
Tests can inject fake LLM clients and do not require Ollama.

## Media Processing and Tool Configuration

Real multimedia adapters are configured in `configs/tools.yaml` and can also be
controlled with environment variables. `RawMediaProcessor` now wires the media
flow end-to-end for every case media asset:

1. metadata inspection through Pillow plus ExifTool and FFprobe when available;
2. scene-aware video keyframe extraction with PySceneDetect and FFmpeg, falling
   back to uniform timestamps;
3. OCR over original images and extracted keyframes;
4. VLM analysis over original images and extracted keyframes through Ollama;
5. basic forensic analysis, including metadata flags and image-level checks;
6. ASR for video audio through FFmpeg and faster-whisper;
7. local reverse-image search over original images and keyframes, backed by
   pHash plus optional OpenCLIP/FAISS visual similarity in the local index.

Relevant media flags in `configs/tools.yaml` include:

```yaml
media:
  enable_ffmpeg_keyframes: true
  keyframe_strategy: scene_detect
  max_keyframes_per_video: 8
  deduplicate_keyframes: true

  enable_vlm_adapter: true
  vlm_provider: ollama
  vlm_model: llava

  enable_ocr_adapter: true
  enable_asr_adapter: true
  enable_forensic_adapter: true
  enable_local_reverse_search: true
  local_reverse_methods: ["phash", "clip_faiss"]
```

Environment overrides are supported for the heavy adapters:

```env
SEMV_ENABLE_VLM=true
SEMV_VLM_PROVIDER=ollama
SEMV_VLM_MODEL=llava
SEMV_ENABLE_OCR=true
SEMV_ENABLE_ASR=true
SEMV_ENABLE_FORENSICS=true
SEMV_ENABLE_LOCAL_REVERSE=true
SEMV_ENABLE_FREE_WEB_SEARCH=false
```

If optional binaries, models, or local services are missing, adapters should emit
synthetic uncertainty evidence instead of crashing. For full local capability,
install FFmpeg/FFprobe for video and audio processing, ExifTool for richer
metadata extraction, EasyOCR model dependencies for OCR, faster-whisper models
for ASR, and an Ollama multimodal model such as `llava` for VLM-based visual
observation. The project implements local pHash plus optional CLIP/FAISS visual
similarity; it does not claim official Google Lens, Yandex, or TinEye reverse
image search integration. Forensics are basic heuristics unless a learned model
is explicitly configured.

| Component | Implemented | Default | Dependency | Notes |
|---|---:|---:|---|---|
| Metadata | yes | on | Pillow, ffprobe, exiftool optional | graceful fallback |
| Keyframes | yes | on | ffmpeg, PySceneDetect optional | scene detection with uniform fallback |
| OCR | yes | configurable | EasyOCR | optional, emits uncertainty if unavailable |
| ASR | yes | configurable | faster-whisper, ffmpeg | optional, video only |
| VLM | yes | configurable | Ollama LLaVA or equivalent | VLM-based visual observation, not ground truth |
| Forensics | basic | on | Pillow/OpenCV | heuristic only |
| Local reverse | yes | on | pHash, optional OpenCLIP/FAISS | local/cached only |
| Web reverse candidate | yes | off unless free web enabled | web search + image compare | not Google Lens |
| Geolocation | partial | configurable | optional cached Nominatim | GPS/clue extraction offline first |
| Escalation | yes | on | none | score/conflict based |

## Parallel Execution

The pipeline can parallelize expensive per-subclaim work. Two environment
variables control this behavior:

```env
SEMV_PARALLEL_DEEP_RESEARCH=true
SEMV_PARALLEL_CLAIMS=true
SEMV_MAX_WORKERS=2
```

`SEMV_PARALLEL_DEEP_RESEARCH` parallelizes deep-research calls across subclaims.
`SEMV_PARALLEL_CLAIMS` parallelizes argument generation, verification, scoring,
QBAF construction, and decision mapping across subclaims. `SEMV_MAX_WORKERS`
limits both parallel sections so local LLM and tool backends are not overloaded.

## Input Format

The current canonical input is `CaseBundle`, defined in
`src/schemas/case_bundle_schema.py`. It contains:

- `dataset`: dataset name, split, native path, format, and adapter version.
- `task`: task type, subtask, media type, output target, and language.
- `input`: title, caption, description, links, location hints, and metadata.
- `claims`: main and scoped subclaims.
- `media_assets`: image, video, audio, screenshot, keyframe, or document assets.
- `source_clusters`: source/uploader/reposter/news/fact-check clusters.
- `temporal_context`: claimed time, publication times, and event-time bounds.
- `location_context`: claimed location, camera/target coordinates, and cues.
- `provided_evidence`: local or dataset-provided evidence records.
- `gold`: hidden or post-prediction labels/reports.
- `run_config`: retrieval, search, memory, contestation, and output controls.

The CLI also accepts legacy `MultimediaCase` JSON files for compatibility, but
new dataset work should use or convert to `CaseBundle`.

## Supported Dataset Adapters

The adapter registry supports:

- `mv2026_folder`
- `cosmos`
- `image_caption`
- `report_style`
- `auto` for adapter auto-detection

Convert a native case to canonical format:

```bash
python scripts/convert_case.py \
  --case-path data/raw/mv2026/training/ID333/ID333 \
  --adapter auto \
  --split training \
  --canonical-root data/canonical
```

## Single-Case Inference

Run a canonical case bundle:

```bash
python scripts/run_case.py \
  --case-bundle data/canonical/mv2026/ID333/case_bundle.json \
  --mode inference_only
```

Run a native dataset case through an adapter:

```bash
python scripts/run_case.py \
  --case-path data/raw/mv2026/training/ID333/ID333 \
  --adapter auto \
  --split training \
  --mode inference_only
```

Run the included local example:

```bash
python scripts/run_case.py \
  --case data/cases/sample_case.json \
  --mode inference_only
```

Supported modes are:

- `inference_only`: prediction and reporting only.
- `self_evolving`: prediction followed by post-prediction reflection.
- `test`: test-safe execution with leakage guards.
- `bootstrap_memory`: post-prediction memory bootstrapping from available gold.



Run the ID333-style MV2026 case with local media adapters enabled:

```bash
SEMV_ENABLE_OCR=true \
SEMV_ENABLE_ASR=true \
SEMV_ENABLE_VLM=true \
SEMV_VLM_MODEL=llava \
SEMV_ENABLE_FORENSICS=true \
SEMV_ENABLE_LOCAL_REVERSE=true \
python scripts/run_case.py \
  --case-path data/raw/mv2026/ID333 \
  --adapter mv2026_folder \
  --split validation \
  --mode inference_only
```

For an offline smoke run with heavy model adapters disabled:

```bash
SEMV_ENABLE_OCR=false \
SEMV_ENABLE_ASR=false \
SEMV_ENABLE_VLM=false \
SEMV_ENABLE_FORENSICS=true \
SEMV_ENABLE_LOCAL_REVERSE=true \
python scripts/run_case.py \
  --case-path data/raw/mv2026/ID333 \
  --adapter mv2026_folder \
  --split validation \
  --mode inference_only
```

A feedback file can be supplied with the current CLI shape:

```bash
python scripts/run_case.py \
  --case-bundle data/canonical/mv2026/ID333/case_bundle.json \
  --mode self_evolving \
  --human-feedback-json data/feedback/ID333_human_feedback.json
```

As noted above, this argument is currently a scaffold for the contestation
integration. Full adaptive re-execution requires wiring the feedback parser into
`run_case_bundle`.

## Outputs

Each run writes artifacts to:

```text
data/outputs/cases/<case_id>/
```

The output directory contains:

```text
input_case_bundle.json
raw_evidence.json
normalized_evidence.json
evidence_graph.json
subclaims.json
arguments.json
qbaf_graphs.json
retrieved_memory.json
report.json
report.md
reflection_candidates.json
verified_memory_updates.json
run_log.txt
```

`report.json` is the structured verification report. `report.md` is the
readable report. Markdown output now includes a dedicated `Media Analysis`
section summarizing metadata, keyframes, OCR, ASR, VLM, forensic, reverse-search,
and geolocation-clue evidence before the generic evidence pool. Intermediate
artifacts are written to support inspection, ablation analysis, human
contestation, and error analysis.

Media-derived working files are written under:

```text
data/outputs/_media/<case_id>/media_<index>/
```

These folders can contain extracted keyframes, ASR audio intermediates, and
forensic outputs such as ELA images. Verify a completed case directory with:

```bash
python scripts/check_case_outputs.py data/outputs/cases/ID333
```

For a full contestation implementation, the recommended additional artifacts are:

```text
contestation_package.json
human_feedback.json
revision_plan.json
contested_report.json
contested_report.md
```

These artifacts should preserve the original prediction and make the human-led
revision auditable rather than overwriting the first-pass output.

## Evaluation Protocols

The joint protocol is configured in `configs/evaluation.yaml`:

```bash
python scripts/run_protocol.py \
  --config configs/evaluation.yaml \
  --output-dir data/outputs/evaluation/joint_mv_cosmos
```

Static dataset evaluators can be run directly:

```bash
python scripts/evaluate_mv2026.py \
  --raw-root data/raw/mv2026/training \
  --output-dir data/outputs/evaluation/mv2026_static \
  --split training \
  --case-id ID333

python scripts/evaluate_cosmos.py \
  --cosmos-metadata data/raw/cosmos/test.jsonl \
  --image-root data/raw/cosmos/images \
  --output-dir data/outputs/evaluation/cosmos_static
```

Evaluation outputs include prediction records, gold records, per-case metrics,
aggregate metrics, confusion matrices, calibration bins, memory metrics, failed
cases, and a Markdown evaluation report.

Implemented metric families include final-label performance, section-wise report
quality, evidence URL checks, geolocation accuracy, temporal accuracy, entity
coverage, report-structure checks, hallucination checks, accuracy, balanced
accuracy, macro F1, AUROC, average precision, expected calibration error, and
Brier score.

## Memory and Self-Evolution

Memory stores are JSONL files under `data/memory/`:

```text
episodic_memory.jsonl
semantic_rules.jsonl
failure_memory.jsonl
```

Seed and consolidate memory with:

```bash
python scripts/seed_memory.py
python scripts/consolidate_memory.py
```

Memory retrieval and updates are controlled by `CaseBundle.run_config` and by
the active evaluation protocol. Protocols can allow retrieval while freezing
updates on held-out splits.

In self-evolving or bootstrap-memory modes, gold labels and reports are still
blocked before prediction. After prediction, the reflection module can compare
prediction behavior against available gold annotations and generate candidate
memory updates. These candidates should be verified before being added to active
memory.

## Reproducibility Notes

- Local LLM behavior depends on the configured Ollama model and decoding
  parameters.
- Web and reverse-search behavior is controlled by each case `run_config` and
  tool configuration. `DeepResearcher` also reuses existing media evidence and
  derives geolocation candidates from OCR, ASR, VLM, metadata, and web clues.
- Parallel execution can change wall-clock runtime but should preserve output
  ordering by claim.
- Gold leakage is checked before pipeline execution.
- Intermediate artifacts are persisted for auditability, ablation studies, and
  contestation review.

## Tests

```bash
pytest
```

The test suite covers case-bundle schemas, ingestion adapters, leakage guards,
claim decomposition, QBAF propagation and scoring, memory retrieval/update
behavior, temporal and geolocation metrics, evaluation adapters, media-derived
evidence handling, and end-to-end report generation.

## Citation

If this repository is used as paper code, cite the accompanying paper or project
release. A BibTeX entry can be added here when the paper metadata is finalized.
