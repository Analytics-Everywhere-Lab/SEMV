# Self-Evolving Contestable QBAF for Multimedia Verification

Reference implementation for a self-evolving, contestable Quantitative Bipolar
Argumentation Framework (QBAF) for multimedia verification. The system verifies
image, video, and image-caption claims by decomposing them into scoped
subclaims, collecting multimodal evidence, constructing support and attack
arguments, propagating argument strength over QBAF graphs, and producing both
machine-readable and human-readable verification reports.

This repository is organized as research code for running single-case inference,
canonical dataset conversion, memory-enabled ablations, and MV2026/COSMOS-style
evaluation protocols.

## Method Overview

For each case, the pipeline performs the following stages:

1. Canonicalize the input as a `CaseBundle`.
2. Load media assets and extract available image, video, OCR, ASR, metadata, and
   keyframe signals.
3. Use provided claims or decompose the main claim into scoped verification
   subclaims.
4. Retrieve relevant verified memory when enabled by the case run configuration.
5. Plan research and optionally run web or reverse-search evidence retrieval.
6. Normalize evidence and construct an evidence graph.
7. Generate, verify, and score support/attack arguments per subclaim.
8. Build and propagate QBAF graphs, resolving clashes when required.
9. Aggregate subclaim decisions into a final verification label and confidence.
10. Optionally reflect after prediction to produce verified memory-update
    candidates.

Gold labels and gold reports are guarded by leakage checks and are only used
post-prediction in self-evolving or bootstrap-memory modes.

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
src/processing/           Media loading, metadata, OCR, ASR, keyframe extraction
src/qbaf/                 QBAF graph construction, propagation, decision mapping
src/reporting/            JSON and Markdown report rendering
tests/                    Unit and integration tests
```

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are intentionally small: `pydantic`, `PyYAML`, `requests`,
`Pillow`, and `pytest`.

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
  --case-path data/raw/mv2026/example_case.json \
  --adapter auto \
  --split validation \
  --canonical-root data/canonical
```

## Single-Case Inference

Run a canonical case bundle:

```bash
python scripts/run_case.py \
  --case-bundle data/canonical/example_case.json \
  --mode inference_only
```

Run a native dataset case through an adapter:

```bash
python scripts/run_case.py \
  --case-path data/raw/mv2026/example_case.json \
  --adapter auto \
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
readable report. Intermediate artifacts are written to support inspection,
ablation analysis, and error analysis.

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
  --raw-root data/raw/mv2026 \
  --output-dir data/outputs/evaluation/mv2026_static \
  --split validation

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

## Memory

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

## Reproducibility Notes

- Local LLM behavior depends on the configured Ollama model and decoding
  parameters.
- Web and reverse-search behavior is controlled by each case `run_config` and
  tool configuration.
- Gold leakage is checked before pipeline execution.
- Intermediate artifacts are persisted for auditability and ablation studies.
- Tests use deterministic fakes where possible.

## Tests

```bash
pytest
```

The test suite covers case-bundle schemas, ingestion adapters, leakage guards,
claim decomposition, QBAF propagation and scoring, memory retrieval/update
behavior, temporal and geolocation metrics, evaluation adapters, and end-to-end
report generation.

## Citation

If this repository is used as paper code, cite the accompanying paper or project
release. A BibTeX entry can be added here when the paper metadata is finalized.
