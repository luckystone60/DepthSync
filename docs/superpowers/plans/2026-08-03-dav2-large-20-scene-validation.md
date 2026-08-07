# DAv2-Large 20-Scene Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the offline V5 photo anchor with DAv2-Large and build a reproducible 20-scene Pexels validation corpus with depth videos and aggregate reporting.

**Architecture:** Keep DepthPro intact as an optional baseline. Add a DAv2 adapter to the isolated model runner, a Pexels API downloader that records immutable source metadata, and a batch orchestration layer that validates every scene through existing VDA, flow, V5, evaluation, and visualization stages. The V5 core and mobile playback contract remain unchanged.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, OpenCV, NumPy, Pexels REST API, existing DepthSync V5 / SEA-RAFT-S tools.

## Global Constraints

- Operate from `C:\Users\jiao\Documents\DepthSync\.worktrees\v5-flow-local-affine` on branch `codex/v5-flow-local-affine`.
- Do not store `PEXELS_API_KEY` in source, configuration, shell history, logs, committed manifests, or PR text.
- DAv2-Large is offline-only and CC-BY-NC-4.0; never add it to any mobile or playback path.
- Input assets, model weights, flow, depth caches, results and videos remain Git-ignored.
- Preserve V4/V5 exact fallback behavior and 3 s / 30 fps / 90 frame validation timeline.
- Use `apply_patch` for all source/document edits; test before each production implementation change; commit each independently testable task.

---

### Task 1: Define DAv2-Large model contract and isolated anchor adapter

**Files:**
- Modify: `tools/run_depth_models.py`
- Modify: `config/model-sources.json`
- Modify: `README.md`
- Test: `tests/test_depth_models.py`

**Interfaces:**
- Consumes: `run_depth_models.py dav2-large --anchor PATH --output-dir PATH --model-id MODEL_ID`.
- Produces: `photo_dav2_large_relative.npy`, `photo_dav2_large_disparity.npy`, `dav2_large.json`.
- Keeps: `run_depthpro()` behavior unchanged.

- [ ] **Step 1: Write failing adapter contract tests**

```python
def test_relative_depth_is_oriented_to_positive_anchor_correlation():
    relative = np.array([[0.0, 1.0]], np.float32)
    vda = np.array([[1.0, 0.0]], np.float32)
    disparity, reversed_direction = orient_relative_depth(relative, vda)
    np.testing.assert_allclose(disparity, [[1.0, 0.0]])
    assert reversed_direction is True

def test_dav2_metadata_contains_model_revision_and_direction(tmp_path):
    write_dav2_outputs(...)
    metadata = json.loads((tmp_path / "dav2_large.json").read_text())
    assert metadata["backend"] == "Depth-Anything-V2-Large"
    assert metadata["direction_reversed"] is False
```

- [ ] **Step 2: Run the focused test to verify failure**

Run: `C:\Users\jiao\Documents\DepthSync\.venv-models\Scripts\python.exe -m pytest tests/test_depth_models.py -q`

Expected: FAIL because `orient_relative_depth` and DAv2 metadata/output functions do not exist.

- [ ] **Step 3: Add pure DAv2 utilities before model loading**

```python
def orient_relative_depth(relative: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, bool]:
    # resize reference if needed, use finite paired pixels and robust Pearson sign
    # reject fewer than 64 valid samples or |correlation| < 0.05
    # return finite FP32 output in the direction positively correlated with VDA disparity

def _write_dav2_outputs(output_dir: Path, relative: np.ndarray, disparity: np.ndarray, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "photo_dav2_large_relative.npy", relative.astype(np.float32))
    np.save(output_dir / "photo_dav2_large_disparity.npy", disparity.astype(np.float32))
    (output_dir / "dav2_large.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Implement `run_dav2_large` and CLI**

Use `AutoImageProcessor` and `AutoModelForDepthEstimation` with the official model ID. Read the anchor image with OpenCV, convert BGR to RGB, execute `torch.inference_mode()`, resize predicted relative depth to source anchor resolution, validate finite values, and call `orient_relative_depth` using an optional `--reference-vda` anchor array. Add `dav2-large` parser arguments `--anchor`, `--output-dir`, `--reference-vda`, `--model-id`, `--revision`, and `--device`. Never include the model cache path in committed config.

- [ ] **Step 5: Add model provenance**

Add `depth_anything_v2_large` to `config/model-sources.json` with repository, model ID, revision resolved at download time, expected license `CC-BY-NC-4.0`, cache destination and a `validation_only: true` flag. Update README with the isolated invocation and license restriction.

- [ ] **Step 6: Run tests and commit**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_depth_models.py -q`

Expected: PASS.

```powershell
git add tools/run_depth_models.py config/model-sources.json README.md tests/test_depth_models.py
git commit -m "feat: add DAv2 Large offline anchor adapter"
```

### Task 2: Implement reproducible Pexels candidate selection and manifest storage

**Files:**
- Create: `depthsync/pexels_dataset.py`
- Create: `tools/fetch_pexels_validation.py`
- Create: `config/pexels-validation-manifest.json`
- Test: `tests/test_pexels_dataset.py`

**Interfaces:**
- Consumes: Pexels API JSON and `PEXELS_API_KEY` environment variable.
- Produces: versioned manifest rows and Git-ignored `testdata/<scene>.mp4` files.
- Main APIs: `select_720p_mp4(video: dict) -> dict`, `validate_manifest(manifest: dict) -> None`, `download_scene(...) -> ManifestRow`.

- [ ] **Step 1: Write failing selection and secret-safety tests**

```python
def test_select_720p_mp4_prefers_exact_short_side_then_smallest_larger_file():
    chosen = select_720p_mp4({"video_files": [...], "duration": 8})
    assert chosen["width"] == 1280 and chosen["height"] == 720

def test_manifest_rejects_duplicate_pexels_ids_and_api_key_text():
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest({"scenes": [row("1"), row("1")]})
    with pytest.raises(ValueError, match="secret"):
        validate_manifest({"scenes": [row("1", query="5JavIZ...")]})
```

- [ ] **Step 2: Run the focused test to verify failure**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_pexels_dataset.py -q`

Expected: FAIL because the dataset module does not exist.

- [ ] **Step 3: Implement pure schema and selection functions**

Define a frozen `ManifestRow` dataclass containing the fields in the approved spec. `select_720p_mp4` accepts only `video/mp4`, requires duration >= 3.0, chooses exact 720 short side first, otherwise the smallest file whose short side is >= 720. `validate_manifest` requires IDs `01`–`20`, unique Pexels IDs, a Pexels page URL, `sha256` with 64 hexadecimal characters for downloaded rows, valid taxonomy tags and no substring matching `PEXELS_API_KEY` or the actual environment secret.

- [ ] **Step 4: Implement API/search/download CLI**

`fetch_pexels_validation.py` reads `os.environ["PEXELS_API_KEY"]`, sends it only as `Authorization` HTTP header, searches the five approved tag query groups, prints candidate metadata without headers, and supports:

```text
--list-candidates --tag walking_turning
--accept <pexels-id> --scene 04 --tag walking_turning
--manifest config/pexels-validation-manifest.json
--output-root testdata
```

On `--accept`, download the chosen MP4 to a temporary sibling path, verify SHA-256 and OpenCV duration, then atomically rename it to `testdata/<scene>.mp4` and update the manifest with source metadata. Do not overwrite an existing scene unless `--replace` is supplied.

- [ ] **Step 5: Seed manifest with existing scenes and category quotas**

Create rows for `01`–`03` with `source_type: local_existing` and empty Pexels fields. Add top-level target counts for five tags and a `schema_version`. Do not invent Pexels metadata for existing local clips.

- [ ] **Step 6: Run tests and commit**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_pexels_dataset.py -q`

Expected: PASS.

```powershell
git add depthsync/pexels_dataset.py tools/fetch_pexels_validation.py config/pexels-validation-manifest.json tests/test_pexels_dataset.py
git commit -m "feat: add reproducible Pexels validation dataset"
```

### Task 3: Add 20-scene preparation and DAv2 anchor batch orchestration

**Files:**
- Create: `tools/run_dav2_validation.py`
- Modify: `depthsync/validation.py`
- Modify: `config/validation-scenes.json`
- Test: `tests/test_validation_batch.py`

**Interfaces:**
- Consumes: valid manifest, 20 source MP4 files, DAv2 output adapter.
- Produces: 20 deterministic clips, anchor PNG paths and DAv2 output directories.
- Main APIs: `scene_ids_from_manifest(path) -> list[str]`, `validate_scene_coverage(...) -> None`.

- [ ] **Step 1: Write failing 20-scene coverage tests**

```python
def test_scene_ids_require_contiguous_01_through_20(tmp_path):
    manifest = write_manifest(tmp_path, scenes=["01", "03"])
    with pytest.raises(ValueError, match="02"):
        scene_ids_from_manifest(manifest)

def test_validation_scene_requires_subject_and_inspection_boxes():
    with pytest.raises(ValueError, match="subject_box"):
        validate_scene_coverage({"04": {"face_box": [0, 0, 1, 1]}}, ["04"])
```

- [ ] **Step 2: Run the focused test to verify failure**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_validation_batch.py -q`

Expected: FAIL because batch coverage helpers do not exist.

- [ ] **Step 3: Implement deterministic scene traversal**

Add helpers that verify exact contiguous scene IDs, existing sources, and a complete `validation-scenes.json` row for every scene. Extend `depthsync.validation` CLI with `--manifest` and `--scenes` while preserving explicit path mode. It must call existing `prepare_validation_clip` and write each anchor frame to `artifacts/clips/<scene>/anchor.png`.

- [ ] **Step 4: Implement DAv2 batch driver**

`tools/run_dav2_validation.py` reads scene IDs, passes each anchor and `artifacts/depth/<scene>/video_disparity.npz` anchor slice to `run_depth_models.py dav2-large`, skips only already valid output triples unless `--force`, records per-scene success/failure in `artifacts/depth/dav2-batch.json`, and returns nonzero after processing if any scene failed.

- [ ] **Step 5: Add annotation template rows**

For scenes `04`–`20`, add JSON template rows only after a candidate is accepted. Each row must contain actual manually measured `face_box`, `subject_box`, `inspection_frames`, `tail_frames` and optional `static_box`; do not use a generic central ROI.

- [ ] **Step 6: Run tests and commit**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_validation_batch.py -q`

Expected: PASS.

```powershell
git add tools/run_dav2_validation.py depthsync/validation.py config/validation-scenes.json tests/test_validation_batch.py
git commit -m "feat: batch DAv2 validation scenes"
```

### Task 4: Route V5 and evaluation to DAv2 anchors without removing DepthPro

**Files:**
- Modify: `depthsync/evaluate.py`
- Modify: `depthsync/depth_visualization.py`
- Modify: `tests/test_evaluate.py`
- Modify: `tests/test_depth_visualization.py`

**Interfaces:**
- Consumes: `--anchor-model dav2-large|depthpro`, default `dav2-large`.
- Produces: V5 DAv2 metrics and `depth_comparison_dav2_v5.mp4`.
- Must fail explicitly if the selected anchor file is missing or invalid.

- [ ] **Step 1: Write failing anchor-selection tests**

```python
def test_dav2_is_the_default_anchor_path(tmp_path):
    assert selected_anchor_path(tmp_path, "dav2-large").name == "photo_dav2_large_disparity.npy"

def test_visualization_uses_fixed_dav2_anchor_range_and_labels(tmp_path):
    metadata = render_scene(..., anchor_model="dav2-large")
    assert metadata["anchor_model"] == "dav2-large"
    assert metadata["files"]["comparison_video"] == "depth_comparison_dav2_v5.mp4"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py tests/test_depth_visualization.py -q`

Expected: FAIL because the selected-anchor API and DAv2 comparison output do not exist.

- [ ] **Step 3: Implement explicit anchor selection**

Add one shared helper (in `depthsync/evaluate.py` or a small new `depthsync/anchors.py`) that maps `dav2-large` and `depthpro` to exact filenames, checks finite values and emits a clear `anchor_invalid` failure. Pass `anchor_model` from both CLIs. Do not silently fall back from DAv2 to DepthPro.

- [ ] **Step 4: Update visualization outputs**

For DAv2 mode, use anchor P02–P98 as the fixed range of V4/V5/anchor panels and label the fourth panel `DAv2-Large photo anchor`. Export `depth_comparison_dav2_v5.mp4`, existing diagnostic video, and `dav2_vs_depthpro_anchor.png` whenever a valid DepthPro baseline file exists. Preserve the V4/V5 comparison format.

- [ ] **Step 5: Run tests and commit**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py tests/test_depth_visualization.py -q`

Expected: PASS.

```powershell
git add depthsync/evaluate.py depthsync/depth_visualization.py tests/test_evaluate.py tests/test_depth_visualization.py
git commit -m "feat: validate V5 against DAv2 anchors"
```

### Task 5: Build 20-scene aggregate reporting and verification commands

**Files:**
- Create: `depthsync/aggregate_validation.py`
- Create: `tests/test_aggregate_validation.py`
- Modify: `README.md`
- Modify: `docs/depthsync-v5-algorithm.md`

**Interfaces:**
- Consumes: 20 `results/dav2/<scene>/metrics.json` files and Pexels manifest.
- Produces: `reports/validation-dav2-20-scenes.md` and `reports/validation-dav2-20-scenes.json`.
- Main API: `aggregate_validation(result_root, manifest_path) -> AggregateReport`.

- [ ] **Step 1: Write failing aggregate tests**

```python
def test_aggregate_reports_tag_median_p90_and_worst_three(tmp_path):
    report = aggregate_validation(result_root, manifest_path)
    assert report.tags["dance_motion"].anchor_nmae_median == pytest.approx(0.02)
    assert [item.scene for item in report.worst_anchor_improvement] == ["17", "05", "03"]

def test_aggregate_lists_missing_or_fallback_scenes_instead_of_averaging_them_away(tmp_path):
    report = aggregate_validation(result_root, manifest_path)
    assert report.incomplete == ["12"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_aggregate_validation.py -q`

Expected: FAIL because the aggregator does not exist.

- [ ] **Step 3: Implement aggregation and CLI**

Load exact manifest scene order. For completed scenes, calculate global and per-tag median/P90 for anchor NMAE, subject anchor NMAE, temporal P95 and fallback count. Sort and report the three lowest anchor improvements and three highest temporal regressions. Include every missing metrics file, invalid DAv2 anchor and V5 fallback in `incomplete`/`exceptions`; never include those rows in a metric denominator without printing the denominator.

- [ ] **Step 4: Document end-to-end commands**

Document the environment-variable setup without embedding the actual key, candidate selection, candidate acceptance, batch preparation, VDA, DAv2, flow, evaluation, visualization and aggregation. State model licensing and that 20-scene results are offline simulation only.

- [ ] **Step 5: Run tests and commit**

Run: `C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_aggregate_validation.py -q`

Expected: PASS.

```powershell
git add depthsync/aggregate_validation.py tests/test_aggregate_validation.py README.md docs/depthsync-v5-algorithm.md
git commit -m "feat: aggregate DAv2 20-scene validation"
```

### Task 6: Acquire, annotate and run the 20-scene offline corpus

**Files:**
- Modify: `config/pexels-validation-manifest.json`
- Modify: `config/validation-scenes.json`
- Create: `reports/validation-dav2-20-scenes.md`

**Interfaces:**
- Consumes: completed code from Tasks 1–5, an authenticated `PEXELS_API_KEY`, ignored MP4/model caches.
- Produces: 20 complete manifest rows, 20 scene annotations, depth/flow/result artifacts and aggregate report.

- [ ] **Step 1: List candidates by tag without downloading**

Run one command per tag:

```powershell
$env:PEXELS_API_KEY = '<provided outside source control>'
C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe tools\fetch_pexels_validation.py --list-candidates --tag walking_turning
```

Inspect source URL, duration, file dimensions, visible full-body/person motion and background complexity. Do not record the environment value in any generated file.

- [ ] **Step 2: Accept 17 non-duplicate candidates**

Use scene IDs `04`–`20`, meeting the tag quota and manual quality checks. For every accepted row run `--accept`, verify the output MP4 opens, verify its hash exists in manifest, and commit only the updated JSON manifest after all 17 rows are complete.

- [ ] **Step 3: Create and verify annotations**

For every new anchor preview, measure actual normalized face/subject boxes, set inspection frames and tail frames, then run:

```powershell
C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest tests/test_validation_batch.py -q
```

Expected: PASS with all 20 scenes complete; no synthetic default ROI.

- [ ] **Step 4: Run all offline models and validation**

Run the existing clips/VDA/flow commands plus DAv2 batch driver for `01`–`20`; then call evaluation and visualization with `--anchor-model dav2-large` and result root `results/dav2`. Preserve processing logs outside Git and stop on invalid DAv2 output rather than substituting DepthPro.

- [ ] **Step 5: Aggregate and manually inspect outputs**

Run `python -m depthsync.aggregate_validation --result-root results/dav2 --manifest config/pexels-validation-manifest.json --report reports/validation-dav2-20-scenes.md`. Play every comparison video; record any wall split, subject halo, occlusion error or jump in the report. Do not claim 20-scene acceptance if any scene is incomplete.

- [ ] **Step 6: Final verification and commit/push**

Run:

```powershell
C:\Users\jiao\Documents\DepthSync\.venv\Scripts\python.exe -m pytest -q
git diff --check
git status --short
```

Commit only code/config/docs/reports/manifest, push `codex/v5-flow-local-affine`, and update the draft PR with the aggregate evidence and offline-only limitation.

## Plan Self-Review

- Spec coverage: Tasks 1 and 4 cover DAv2 replacement and DepthPro comparison; Tasks 2, 3 and 6 cover 17 Pexels downloads, 20-scene completion and manual ROI; Task 5 covers aggregate evidence; Task 6 covers real assets and subjective review.
- Placeholder scan: no implementation placeholders are used; model revision is resolved and recorded at execution rather than falsely hard-coded before download.
- Type consistency: `ManifestRow`, `select_720p_mp4`, `validate_manifest`, `scene_ids_from_manifest`, `selected_anchor_path`, and `aggregate_validation` are declared with their consumers in the relevant tasks.
