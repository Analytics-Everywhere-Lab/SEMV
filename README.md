# Self-Evolving Contestable A-QBAF for Multimedia Verification

Reference implementation for a self-evolving, contestable Arena-based Quantitative Bipolar
Argumentation Framework (A-QBAF) for multimedia verification. The system verifies
image, video, and image-caption claims by decomposing them into scoped
subclaims, collecting multimodal evidence, constructing support and attack
arguments, propagating argument strength over A-QBAF graphs, and producing both
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
8. Build and propagate A-QBAF graphs, resolving clashes when required.
9. Aggregate subclaim decisions into a final verification label and confidence.
10. Render structured JSON and Markdown reports with evidence, media analysis,
    argument, A-QBAF, uncertainty, and memory traces.
11. Optionally reflect after prediction to produce verified memory-update
    candidates.

Gold labels and gold reports are guarded by leakage checks and are only used
post-prediction in self-evolving or bootstrap-memory modes.

## Human Contestation and Adaptive Revision

SEMV is intended to support contestable verification rather than a closed
one-shot prediction. The human reviewer does not need to directly edit every
internal pipeline artifact. Instead, the contestation interface is centered on
arguments, because arguments are the bridge between evidence, subclaims, A-QBAF
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
where revision is needed from the action type and the free-text `reason`.
Each contestation can also include an explicit `metadata.revision_target`
field to force which stage should be revisited, overriding the inferred
routing.

Valid `revision_target` / `rerun_from_step` values (see
`src/schemas/contestation_schema.py::RevisionTarget`) are:

```text
claim_decomposition
evidence_retrieval
evidence_validation
argument_construction
qbaf_reasoning
final_aggregation
report_generation
```

This makes the contestation mechanism adaptive. For example, rejecting an
argument because the retrieved source does not actually support the claim
sends the pipeline back to `evidence_retrieval`, not merely rescore the
argument. Rejecting an argument because the evidence itself does not support
the claim (wrong date, wrong location, wrong entity, irrelevant source)
restarts from `evidence_validation`. Editing an argument's wording or adding a
new argument backed by existing evidence restarts from
`argument_construction`. Accepting an argument, or rejecting/editing it for
reasons unrelated to evidence, only reruns `qbaf_reasoning` and the final
report. See `src/contestation/revision_router.py` and
`src/contestation/adaptive_revision_executor.py` for the full routing and
rerun implementation.

A typical feedback file can use the following shape:

```json
{
  "case_id": "ID333",
  "reviewer_id": "human_1",
  "contestations": [
    {
      "contestation_id": "c1",
      "case_id": "ID333",
      "action": "reject",
      "target_argument_id": "arg_where_002",
      "reason": "The cited source describes a similar scene but not the claimed location."
    },
    {
      "contestation_id": "c2",
      "case_id": "ID333",
      "action": "edit",
      "target_argument_id": "arg_when_001",
      "edited_text": "The evidence supports that the video was online by the publication date, but not the exact recording date.",
      "edited_stance": "attack",
      "edited_confidence": 0.72
    },
    {
      "contestation_id": "c3",
      "case_id": "ID333",
      "action": "add",
      "added_subclaim_id": "authenticity_1",
      "added_stance": "support",
      "added_text": "No visible editing artifacts are reported by the available forensic evidence.",
      "added_evidence_ids": ["ev_forensic_001"]
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
- `--human-feedback-json` is accepted by `scripts/run_case.py` as an alias for
- `--human_review_path` and is passed into `run_case_bundle`, which routes the
contestation batch via `route_revision`, reruns only the affected pipeline
stage and its downstream steps via `execute_adaptive_revision`, and writes the
before/after contestation artifacts alongside the final report.

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
src/A-QBAF/                 A-QBAF graph construction, propagation, decision mapping
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

All agent-like components share `VLLMOpenAIClient`, built via `build_llm_client()`.
SEMV talks to a local vLLM server through its OpenAI-compatible
`/v1/chat/completions` endpoint. Configure it in `.env`:

```env
SEMV_LLM_PROVIDER=vllm

VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen/Qwen3.5-9B

VLLM_TEMPERATURE=0.0
VLLM_TOP_P=1.0
VLLM_TOP_K=20
VLLM_MAX_TOKENS=4096
VLLM_TIMEOUT=120

# For SEMV JSON-heavy pipeline, disable model thinking in requests.
VLLM_ENABLE_THINKING=false

# Use same model for visual analysis.
SEMV_VLM_PROVIDER=vllm
SEMV_VLM_MODEL=Qwen/Qwen3.5-9B
```

`VLLM_MODEL` must name a model already served by the local vLLM server. Tests
can inject fake LLM clients and do not require vLLM.

Start the server before running the pipeline. It is recommended to keep vLLM
in a separate environment from the rest of SEMV's dependencies, since vLLM
pulls in heavy CUDA/Torch requirements:

```bash
uv pip install vllm --torch-backend=auto
```

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90
```

Context length (`--max-model-len`) is configured at server startup rather than
per request. Start with `32768` to reduce OOM risk and increase to `65536` or
higher only if the GPU has enough memory. Do not pass `--language-model-only`
when `SEMV_ENABLE_VLM=true`, since SEMV needs image/frame analysis; use it
only for text-only runs with `SEMV_ENABLE_VLM=false`.

Quick server check:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-9B",
    "messages": [
      {"role": "user", "content": "Return JSON only: {\"ok\": true}"}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Media Processing and Tool Configuration

Real multimedia adapters are configured in `configs/tools.yaml` and can also be
controlled with environment variables. `RawMediaProcessor` now wires the media
flow end-to-end for every case media asset:

1. metadata inspection through Pillow plus ExifTool and FFprobe when available;
2. scene-aware video keyframe extraction with PySceneDetect and FFmpeg, falling
   back to uniform timestamps;
3. OCR over original images and extracted keyframes;
4. VLM analysis over original images and extracted keyframes through vLLM;
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
  vlm_provider: vllm
  vlm_model: Qwen/Qwen3.5-9B

  enable_ocr_adapter: true
  enable_asr_adapter: true
  enable_forensic_adapter: true
  enable_local_reverse_search: true
  local_reverse_methods: ["phash", "clip_faiss"]
```

Environment overrides are supported for the heavy adapters:

```env
SEMV_ENABLE_VLM=true
SEMV_VLM_PROVIDER=vllm
SEMV_VLM_MODEL=Qwen/Qwen3.5-9B
SEMV_ENABLE_OCR=true
SEMV_ENABLE_ASR=true
SEMV_ENABLE_FORENSICS=true
SEMV_ENABLE_LOCAL_REVERSE=true
SEMV_ENABLE_FREE_WEB_SEARCH=false
```

### Optional GDELT live news retrieval

SEMV can optionally query [GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
for live news-style evidence. It runs as a separate retrieval step in
`DeepResearcher`, after cached/manual evidence and before the DuckDuckGo-backed
`FreeWebSearch` fallback:

```text
1. Existing case/manual/cached evidence
2. Existing local/cached adapters
3. GDELT live news search
4. DuckDuckGo free web search fallback
5. Synthetic uncertainty item if nothing is found
```

It is disabled by default and does not replace `FreeWebSearch`. Enable it with:

```bash
SEMV_ENABLE_GDELT_SEARCH=true \
python scripts/run_case.py ...
```

Relevant flags in `configs/tools.yaml` (`retrieval:` section):

```yaml
  gdelt_search_enabled: false
  gdelt_base_url: "https://api.gdeltproject.org/api/v2/doc/doc"
  gdelt_max_records_per_claim: 8
  gdelt_max_queries_per_claim: 4
  gdelt_timespan: "3months"
  gdelt_source_lang: "eng"
  gdelt_cache_enabled: true
  gdelt_cache_path: "data/cache/gdelt_search_cache.json"
  gdelt_min_interval_sec: 12
  gdelt_max_retries: 3
  gdelt_backoff_base_sec: 20
  gdelt_circuit_breaker_cooldown_sec: 300
```

Environment overrides:

```env
SEMV_ENABLE_GDELT_SEARCH=true
SEMV_GDELT_TIMESPAN=3months
SEMV_GDELT_SOURCE_LANG=eng
SEMV_GDELT_MAX_QUERIES_PER_CLAIM=1
SEMV_GDELT_MAX_RECORDS_PER_CLAIM=3
SEMV_GDELT_MIN_INTERVAL_SEC=12
SEMV_GDELT_MAX_RETRIES=3
SEMV_GDELT_BACKOFF_BASE_SEC=20
```

Results are represented as `EvidenceItem(source_type="news_article")` with
`provenance.retrieval_method="gdelt_doc_api_artlist"`, preserving article URL,
title, domain, language, source country, and seen date. Responses are cached
in `data/cache/gdelt_search_cache.json`; for reproducible evaluation, run once
with GDELT enabled to populate the cache, then keep the cache fixed for
benchmark reporting.

**Rate limiting and 429 handling.** GDELT throttles the DOC API aggressively,
so SEMV protects itself in a few layers:

- **Query quality filter**: generic single-concept queries (`location`,
  `where`, `image`, `event`, ...) or queries with fewer than two meaningful
  tokens are never sent to GDELT. If no specific, entity/event-rich query can
  be built for a claim, GDELT is skipped entirely for that claim and
  cached/local/DuckDuckGo retrieval still runs.
- **Per-instance rate limiting**: consecutive live requests are spaced at
  least `gdelt_min_interval_sec` apart (default 12s).
- **Retry with backoff**: on `HTTP 429`, SEMV retries up to
  `gdelt_max_retries` times, honoring the `Retry-After` header when present or
  falling back to exponential backoff (`gdelt_backoff_base_sec * 2^attempt`).
- **Circuit breaker**: if retries are exhausted while still rate-limited,
  GDELT is disabled for `gdelt_circuit_breaker_cooldown_sec` (default 300s)
  for the rest of the run; other retrieval sources are unaffected.

For a conservative first run, start with:

```bash
SEMV_ENABLE_GDELT_SEARCH=true \
SEMV_GDELT_MAX_QUERIES_PER_CLAIM=1 \
SEMV_GDELT_MAX_RECORDS_PER_CLAIM=3 \
SEMV_GDELT_TIMESPAN=1month \
python scripts/run_case.py ...
```

## Deep Forensics / TruFor Setup

`ForensicAnalyzer` supports two forensic engines, selected by `forensic_engine`
(`configs/tools.yaml`) or `SEMV_FORENSIC_ENGINE`:

- `basic` (default): no extra install required. Runs the built-in ELA, noise,
  blur, and metadata-flag heuristics entirely inside SEMV.
- `trufor` / `deep`: runs a pluggable deep-forensics backend
  (`src/processing/deep_forensics/`) that produces a pixel-level anomaly map,
  a confidence map, and a whole-image manipulation score. The default backend
  is [TruFor](https://github.com/grip-unina/TruFor), selected via
  `forensic_deep_backend` (`SEMV_FORENSIC_DEEP_BACKEND`, default `trufor`).

Using `basic` needs no additional setup. Using `trufor` requires installing
TruFor separately, since it is not a pip-installable package and ships its own
heavy dependency stack (its own PyTorch/CUDA environment):

```bash
mkdir -p external
git clone https://github.com/grip-unina/TruFor.git external/TruFor
cd external/TruFor/TruFor_train_test
conda env create -f trufor_conda.yaml
conda activate trufor
```

Then download the TruFor pretrained weights and place them at:

```text
external/TruFor/TruFor_train_test/pretrained_models/trufor.pth.tar
```

Run SEMV with deep forensics enabled:

```env
SEMV_ENABLE_FORENSICS=true
SEMV_FORENSIC_ENGINE=trufor
SEMV_FORENSIC_DEEP_BACKEND=trufor
SEMV_TRUFOR_REPO_DIR=external/TruFor/TruFor_train_test
SEMV_TRUFOR_WEIGHTS=external/TruFor/TruFor_train_test/pretrained_models/trufor.pth.tar
# Optional: point at the interpreter inside the TruFor conda env, since it has
# a separate dependency stack from the rest of SEMV.
SEMV_TRUFOR_PYTHON=/path/to/miniconda3/envs/trufor/bin/python
SEMV_TRUFOR_TIMEOUT_SEC=300
SEMV_TRUFOR_EXPERIMENT=trufor_ph3
```

SEMV invokes TruFor's own `test.py` as a subprocess (not a direct Python
import), so the TruFor environment does not need to match SEMV's environment
as long as `SEMV_TRUFOR_PYTHON` points at an interpreter that can run it.

Additional forensic tuning keys in `configs/tools.yaml`:

```yaml
media:
  forensic_max_targets: 8
  forensic_manipulation_threshold: 0.50
  forensic_min_confidence: 0.30
  forensic_save_maps: true
  forensic_fallback_to_basic: true
```

`forensic_save_maps: false` skips writing anomaly/confidence/overlay PNGs to
disk while still computing and reporting the manipulation score and map
statistics. `forensic_fallback_to_basic` controls what happens if the TruFor
repo, weights, or subprocess call are unavailable or fail on every target: if
`true`, SEMV falls back to the `basic` engine and tags the resulting evidence
with `deep_forensic_backend_unavailable` / `deep_forensic_inference_failed`
and `deep_forensic_fallback`; if `false`, SEMV returns a `synthetic_uncertainty`
evidence item with the same flag instead of silently degrading.

**Without `external/TruFor` cloned and real weights downloaded, `trufor` engine
runs will not find the repo/weights and SEMV falls back to basic forensics**
(or returns a `synthetic_uncertainty` item if fallback is disabled) — this is
expected until TruFor is installed as described above.

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
A-QBAF construction, and decision mapping across subclaims. `SEMV_MAX_WORKERS`
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
  --case-path data/raw/mv2026/training/ID333 \
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

For an offline smoke run with heavy model adapters enabled:

```bash
SEMV_ENABLE_GDELT_SEARCH=true \
SEMV_ENABLE_OCR=true \
SEMV_ENABLE_ASR=true \
SEMV_ENABLE_VLM=true \
SEMV_ENABLE_FORENSICS=true \
SEMV_ENABLE_LOCAL_REVERSE=true \
python scripts/run_case.py \
  --case-path data/raw/mv2026/training/ID333 \
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

The same feedback file can also be supplied with `--human_review_path`; both
flags feed the same contestation path.

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

- Local LLM behavior depends on the configured vLLM model and decoding
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
claim decomposition, A-QBAF propagation and scoring, memory retrieval/update
behavior, temporal and geolocation metrics, evaluation adapters, media-derived
evidence handling, and end-to-end report generation.

## Citation

If this repository is used as paper code, cite the accompanying paper or project
release. A BibTeX entry can be added here when the paper metadata is finalized.
