# DepthSync V5 Local Affine Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a flow-guided local affine correction field that improves regional alignment to the photo depth while preserving V4 temporal stability and keeping playback model-free.

**Architecture:** Preserve the current V4 global monotonic LUT as the base result. Estimate an overlapping-window local affine field at the photo anchor, propagate only its scale/offset/confidence through bidirectional dense flow, regularize the field at `64 × 36`, and apply it with depth-guided `2 × 2` upsampling. Any unavailable or unreliable V5 data resolves to the identity correction, so the result exactly falls back to V4.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, pytest/unittest-compatible tests, PyTorch 2.2-compatible SEA-RAFT-S reference adapter, NPZ diagnostics, MP4 visualization.

## Global Constraints

- Work in an isolated Git worktree on branch `codex/v5-flow-local-affine`; keep every task as a separate revertible commit.
- Validate only the existing centered 3-second, 90-frame clips for scenes `01`, `02`, and `03`.
- The photo anchor index remains frame `45`; depth convention remains normalized disparity with larger values nearer.
- The primary local field is `64 × 36 × (delta_scale, offset_norm, confidence)` in FP16, approximately `1.19 MiB` for 90 frames.
- Clamp final local scale to `[0.5, 1.5]`, normalized offset to `[-0.35, 0.35]`, per-frame scale change to `0.03`, and per-frame normalized offset change to `0.02`.
- Dense flow runs only during the post-capture preparation stage, at a default RGB input size of `256 × 144`; playback runs no neural network.
- Use continuous confidence only. Do not introduce hard person masks, fixed region labels, propagated photo residuals, or parameter copying into occluded/revealed pixels.
- If flow inference, field fitting, deserialization, or confidence validation fails, V5 must return the V4 output rather than a partially valid correction.
- Use one fixed photo-derived display range for every frame of each comparison video; never normalize visualization per frame.
- The Python prototype reports measured preparation time, apply time, and peak parameter bytes without claiming the phone target is met.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `depthsync/local_field.py` | V5 field data structures, anchor fitting, confidence propagation, regularization, temporal filtering, playback upsampling, serialization |
| `depthsync/flow.py` | Dense flow data contract, bidirectional consistency confidence, scene-cut isolation, NPZ I/O |
| `depthsync/sea_raft.py` | Optional SEA-RAFT-S adapter and RGB video inference; no dependency imported at package import time |
| `depthsync/core.py` | Preserve V4 base pipeline; dispatch V5 preparation and attach field data to `SyncResult` |
| `depthsync/evaluate.py` | Run V4/V5 baselines, compute hard-gate metrics, save V5 depth and parameter artifacts |
| `depthsync/depth_visualization.py` | Generate four-panel depth comparison, four-panel diagnostics, and specified PNG frames |
| `tools/run_flow_model.py` | CLI for cached SEA-RAFT flow generation and per-stage timing |
| `config/model-sources.json` | Pin SEA-RAFT source, model identifier, local checkpoint path, and verified hash |
| `config/validation-scenes.json` | Add exact evaluation ROIs and frame-specific inspection crops |
| `tests/test_local_field.py` | Synthetic local fit, occlusion, edge, fallback, and round-trip tests |
| `tests/test_flow.py` | Flow direction, consistency, uncertainty, bounds, cut, and NPZ tests |
| `tests/test_depthsync.py` | V5 integration and exact V4 fallback tests |
| `tests/test_evaluate.py` | Metric and gate logic tests |
| `README.md` | V5 preparation, playback, weight setup, validation, and artifact commands |
| `docs/depthsync-flow.drawio` | Updated V5 preparation/playback flow |
| `docs/depthsync-flow.png` | Rendered Draw.io preview |
| `reports/validation-v5.md` | Final measured comparison and pass/fail table |

---

### Task 1: Define the V5 Field Contract and Exact Identity Playback

**Files:**
- Create: `depthsync/local_field.py`
- Create: `tests/test_local_field.py`
- Modify: `depthsync/__init__.py`

**Interfaces:**
- Consumes: a V4 base disparity frame `base: np.ndarray[H,W]`.
- Produces: `LocalFieldConfig`, `LocalFieldSequence`, `apply_local_field(base, field_frame, config)`, `save_local_fields(path, fields)`, and `load_local_fields(path)`.

- [ ] **Step 1: Write failing contract and identity tests**

```python
def test_zero_confidence_is_bit_exact_identity():
    base = np.linspace(0.1, 0.9, 35, dtype=np.float32).reshape(5, 7)
    fields = LocalFieldSequence(
        delta_scale=np.full((1, 3, 4), 0.4, np.float32),
        offset_norm=np.full((1, 3, 4), 0.2, np.float32),
        confidence=np.zeros((1, 3, 4), np.float32),
        depth_low=np.zeros((1, 3, 4), np.float32),
        photo_range=0.8,
    )
    actual = apply_local_field(base, fields.frame(0), LocalFieldConfig(grid_shape=(3, 4)))
    np.testing.assert_array_equal(actual, base)


def test_field_round_trip_preserves_fp16_payload(tmp_path):
    shape = (3, 3, 4)
    fields = LocalFieldSequence(
        delta_scale=np.linspace(-0.1, 0.1, np.prod(shape), dtype=np.float32).reshape(shape),
        offset_norm=np.linspace(-0.2, 0.2, np.prod(shape), dtype=np.float32).reshape(shape),
        confidence=np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape),
        depth_low=np.full(shape, 0.5, np.float32),
        photo_range=0.8,
    )
    path = tmp_path / "v5_fields.npz"
    save_local_fields(path, fields)
    loaded = load_local_fields(path)
    np.testing.assert_array_equal(loaded.delta_scale, fields.delta_scale.astype(np.float16))
    np.testing.assert_array_equal(loaded.offset_norm, fields.offset_norm.astype(np.float16))
    np.testing.assert_array_equal(loaded.confidence, fields.confidence.astype(np.float16))
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_field.py -v
```

Expected: collection fails because `depthsync.local_field` does not exist.

- [ ] **Step 3: Implement immutable contracts, validation, identity fast path, and NPZ I/O**

```python
@dataclass(frozen=True)
class LocalFieldConfig:
    grid_shape: tuple[int, int] = (36, 64)
    scale_bounds: tuple[float, float] = (0.5, 1.5)
    offset_bound_fraction: float = 0.35
    min_fit_pixels: int = 24
    scale_ridge: float = 1e-2
    offset_ridge: float = 1e-3
    spatial_iterations: int = 5
    rgb_sigma: float = 24.0
    depth_sigma_fraction: float = 0.04
    spatial_weight: float = 2.0
    identity_weight: float = 1.0
    temporal_smoothing: float = 0.65
    max_scale_step: float = 0.03
    max_offset_step_fraction: float = 0.02
    upsample_depth_sigma_fraction: float = 0.04
    eps: float = 1e-6


@dataclass(frozen=True)
class LocalFieldFrame:
    delta_scale: np.ndarray
    offset_norm: np.ndarray
    confidence: np.ndarray
    depth_low: np.ndarray
    photo_range: float


@dataclass(frozen=True)
class LocalFieldSequence:
    delta_scale: np.ndarray
    offset_norm: np.ndarray
    confidence: np.ndarray
    depth_low: np.ndarray
    photo_range: float

    def frame(self, index: int) -> LocalFieldFrame:
        return LocalFieldFrame(
            self.delta_scale[index],
            self.offset_norm[index],
            self.confidence[index],
            self.depth_low[index],
            self.photo_range,
        )
```

`apply_local_field` must first validate shapes and finite metadata, then return `base.copy()` before interpolation when `max(confidence) <= 0`. Serialize the three persisted channels as FP16 and metadata as scalar/fixed-size arrays. `depth_low` is preparation-time diagnostic data and remains FP16 in the prototype NPZ; exclude it from the stated mobile persistent byte count.

- [ ] **Step 4: Run tests and the existing suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_field.py tests/test_depthsync.py -v
```

Expected: all tests pass; existing V4 behavior is unchanged.

- [ ] **Step 5: Commit the contract**

```powershell
git add depthsync/local_field.py depthsync/__init__.py tests/test_local_field.py
git commit -m "feat: define V5 local field playback contract"
```

---

### Task 2: Fit an Overlapping-Window Local Affine Field at the Anchor

**Files:**
- Modify: `depthsync/local_field.py`
- Modify: `tests/test_local_field.py`

**Interfaces:**
- Consumes: `fit_anchor_field(base_anchor, photo_anchor, config)`.
- Produces: `LocalFieldFrame` with continuous scale, offset, confidence, and downsampled base depth.

- [ ] **Step 1: Add failing same-value/different-target and insufficient-support tests**

```python
def test_local_fit_solves_regions_global_mapping_cannot_separate():
    h, w = 72, 128
    base = np.full((h, w), 0.45, np.float32)
    photo = base.copy()
    photo[:, : w // 2] += 0.12
    photo[:, w // 2 :] -= 0.08
    field = fit_anchor_field(base, photo, LocalFieldConfig(grid_shape=(18, 32)))
    output = apply_local_field(base, field, LocalFieldConfig(grid_shape=(18, 32)))
    global_output = base + np.median(photo - base)
    assert np.mean(np.abs(output - photo)) < 0.5 * np.mean(np.abs(global_output - photo))


def test_invalid_window_returns_identity_with_zero_confidence():
    base = np.full((24, 32), np.nan, np.float32)
    photo = np.ones((24, 32), np.float32)
    field = fit_anchor_field(base, photo, LocalFieldConfig(grid_shape=(6, 8)))
    assert not np.any(field.delta_scale)
    assert not np.any(field.offset_norm)
    assert not np.any(field.confidence)
```

- [ ] **Step 2: Run the two tests and confirm missing fit behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_field.py -k "local_fit or invalid_window" -v
```

Expected: fail because `fit_anchor_field` is absent.

- [ ] **Step 3: Implement robust overlapping fits**

For each grid center, use a window of `max(9, round(2.5 * H/Gh)) × max(9, round(2.5 * W/Gw))`. Solve weighted ridge regression around identity:

```python
lhs = design.T @ (weights[:, None] * design)
lhs += np.diag([config.scale_ridge, config.offset_ridge])
rhs = design.T @ (weights * target)
scale, offset = np.linalg.solve(lhs, rhs)
```

Use two Tukey-reweighting iterations. Compute confidence as the product of valid support ratio, `exp(-median_abs_residual / (0.08 * photo_range))`, and the clipped inverse condition score. Enforce a minimum of 24 valid pixels per window, then clamp scale/offset to the global constraints. Apply a separable `[1, 2, 1]` confidence-weighted smoothing pass to the parameter grid; never fill a zero-confidence grid point from its neighbors during this anchor-fit step.

- [ ] **Step 4: Run the complete local-field tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_field.py -v
```

Expected: all tests pass, including identity and serialization.

- [ ] **Step 5: Commit anchor fitting**

```powershell
git add depthsync/local_field.py tests/test_local_field.py
git commit -m "feat: fit robust anchor local affine fields"
```

---

### Task 3: Define Dense Bidirectional Flow and Reliability

**Files:**
- Create: `depthsync/flow.py`
- Create: `tests/test_flow.py`

**Interfaces:**
- Produces `DenseFlowSequence`, `compute_flow_confidence`, `detect_scene_cuts`, `save_flow`, and `load_flow`.
- `to_next[p,y,x]` maps frame `p` pixels to frame `p+1`; `to_previous[p,y,x]` maps frame `p+1` pixels to frame `p`. Both arrays have `T-1` entries and use pixel units at the stored flow resolution.

- [ ] **Step 1: Write failing direction, occlusion, cut, and round-trip tests**

```python
def test_forward_backward_error_rejects_inconsistent_strip():
    forward = np.zeros((8, 12, 2), np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = 2.0
    backward[..., 0] = -2.0
    backward[:, 5:7, 0] = 0.0
    confidence = compute_flow_confidence(forward, backward, np.ones((8, 12), np.float32))
    assert np.median(confidence[:, 5:7]) < 0.1
    assert np.median(confidence[:, :3]) > 0.8


def test_scene_cut_blocks_both_directions():
    black = np.zeros((16, 24, 3), np.uint8)
    white = np.full_like(black, 255)
    cuts = detect_scene_cuts([black, black, white, white])
    np.testing.assert_array_equal(cuts, [False, True, False])
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_flow.py -v
```

Expected: collection failure for `depthsync.flow`.

- [ ] **Step 3: Implement the dense flow contract and confidence equation**

```python
@dataclass(frozen=True)
class DenseFlowSequence:
    to_next: np.ndarray
    to_previous: np.ndarray
    confidence_next: np.ndarray
    confidence_previous: np.ndarray
    scene_cuts: np.ndarray
    frame_shape: tuple[int, int]
```

Compute:

```python
back_at_forward = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR)
fb_error = np.linalg.norm(forward + back_at_forward, axis=-1)
tau = 1.5 + 0.01 * np.linalg.norm(forward, axis=-1)
confidence = model_confidence * np.exp(-np.square(fb_error / tau))
confidence *= in_bounds.astype(np.float32)
```

Add a robust photometric factor from grayscale and Sobel-gradient differences, clipped so exposure changes lower confidence continuously rather than creating a binary mask. Set confidence on both sides of a detected scene cut to zero. Store all flow and confidence arrays as FP16 in NPZ with direction names and frame shape.

- [ ] **Step 4: Run flow tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_flow.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit dense flow reliability**

```powershell
git add depthsync/flow.py tests/test_flow.py
git commit -m "feat: add bidirectional flow reliability"
```

---

### Task 4: Propagate and Regularize the Local Field Without Occlusion Leakage

**Files:**
- Modify: `depthsync/local_field.py`
- Modify: `tests/test_local_field.py`

**Interfaces:**
- Consumes: `propagate_local_fields(base_depths, rgb_frames, anchor_field, anchor_index, flow, config)`.
- Produces: a `LocalFieldSequence` covering all frames.

- [ ] **Step 1: Add failing moving-foreground, edge, and temporal-clamp tests**

```python
def test_config():
    return LocalFieldConfig(grid_shape=(18, 32), spatial_iterations=5)


def zero_flow(frame_count, grid_shape):
    gh, gw = grid_shape
    pair_shape = (frame_count - 1, gh, gw)
    return DenseFlowSequence(
        to_next=np.zeros(pair_shape + (2,), np.float32),
        to_previous=np.zeros(pair_shape + (2,), np.float32),
        confidence_next=np.ones(pair_shape, np.float32),
        confidence_previous=np.ones(pair_shape, np.float32),
        scene_cuts=np.zeros(frame_count - 1, bool),
        frame_shape=grid_shape,
    )


def translating_foreground_fixture():
    cfg = test_config()
    gh, gw = cfg.grid_shape
    base = np.full((3, gh, gw), 0.4, np.float32)
    rgb = np.zeros((3, gh, gw, 3), np.uint8)
    delta = np.zeros((gh, gw), np.float32)
    offset = np.zeros_like(delta)
    confidence = np.zeros_like(delta)
    offset[5:15, 5:9] = 0.15
    confidence[5:15, 5:9] = 1.0
    anchor = LocalFieldFrame(delta, offset, confidence, base[1], 1.0)
    flow = zero_flow(3, (gh, gw))
    flow.to_next[1, ..., 0] = 4.0
    flow.to_previous[1, ..., 0] = -4.0
    return base, rgb, anchor, flow


def flat_wall_fixture():
    cfg = test_config()
    gh, gw = cfg.grid_shape
    base = np.full((3, gh, gw), 0.4, np.float32)
    rgb = np.full((3, gh, gw, 3), 128, np.uint8)
    offset = np.tile(
        np.where(np.arange(gw) % 2 == 0, -0.04, 0.04).astype(np.float32),
        (gh, 1),
    )
    anchor = LocalFieldFrame(
        np.zeros_like(offset), offset, np.ones_like(offset), base[1], 1.0
    )
    return base, rgb, anchor, zero_flow(3, (gh, gw))


def test_revealed_background_falls_back_instead_of_inheriting_foreground():
    base, rgb, anchor_field, flow = translating_foreground_fixture()
    fields = propagate_local_fields(base, rgb, anchor_field, 1, flow, test_config())
    revealed = np.s_[2, 5:15, 5:9]
    assert np.max(fields.confidence[revealed]) < 0.05
    assert np.max(np.abs(fields.offset_norm[revealed])) < 1e-6


def test_flat_background_has_no_grid_boundary_steps():
    base, rgb, anchor_field, flow = flat_wall_fixture()
    fields = propagate_local_fields(base, rgb, anchor_field, 1, flow, test_config())
    correction = fields.offset_norm[0]
    assert np.quantile(np.abs(np.diff(correction, axis=1)), 0.99) < 0.01


def test_reliable_trajectory_respects_per_frame_parameter_clamps():
    previous = np.zeros((3, 4, 2), np.float32)
    candidate = np.ones_like(previous)
    clamped = clamp_field_step(previous, candidate, test_config())
    assert np.max(np.abs(clamped[..., 0])) <= 0.030001
    assert np.max(np.abs(clamped[..., 1])) <= 0.020001
```

- [ ] **Step 2: Run focused tests and confirm propagation is absent**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_field.py -k "revealed or boundary or trajectory" -v
```

Expected: fail because propagation and step clamping are not implemented.

- [ ] **Step 3: Implement bidirectional propagation**

Resize flow to the field grid while scaling vector magnitudes to grid pixels. Starting at the anchor and walking independently toward both ends:

```python
warped_q = remap(previous_q, current_to_previous)
warped_c = remap(previous_c, current_to_previous)
data_c = np.clip(warped_c * flow_confidence, 0.0, 1.0)
q = data_c[..., None] * warped_q
```

Pixels with `data_c <= 1e-3` must remain exactly `(delta_scale, offset_norm, confidence)=(0,0,0)` before and after regularization.

- [ ] **Step 4: Implement five edge-aware Jacobi iterations and two-pass trajectory filtering**

Use four-neighbor weights:

```python
pair_weight = np.exp(
    -np.abs(rgb_delta) / config.rgb_sigma
    -np.abs(depth_delta) / (config.depth_sigma_fraction * photo_range)
)
```

The Jacobi numerator combines the warped data term, valid-neighbor smoothness terms, and an identity prior weighted by `1-data_c`. Do not allow neighbor terms to activate zero-confidence occlusion pixels; their confidence remains zero. After spatial regularization, run forward and backward exponential filtering along flow trajectories, then apply the per-frame scale and offset clamps from the global constraints.

- [ ] **Step 5: Run all synthetic field and flow tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_field.py tests/test_flow.py -v
```

Expected: all tests pass; revealed background remains exact identity.

- [ ] **Step 6: Commit propagation**

```powershell
git add depthsync/local_field.py tests/test_local_field.py
git commit -m "feat: propagate edge-safe V5 local fields"
```

---

### Task 5: Integrate V5 With the Existing V4 Base Pipeline

**Files:**
- Modify: `depthsync/core.py`
- Modify: `depthsync/__init__.py`
- Modify: `tests/test_depthsync.py`

**Interfaces:**
- Extend `DepthSyncConfig` with V5 parameters and `algorithm_version="v5"` support.
- Extend `DepthSync.offline_prepare(..., rgb_frames: Sequence[np.ndarray] | None = None, dense_flow: DenseFlowSequence | None = None)`.
- Extend `SyncResult` with `local_fields: LocalFieldSequence | None`.
- Extend `FrameParameters` with `local_field: LocalFieldFrame | None`.
- `DepthSync.apply_frame` applies V4 first, then V5 when a field frame is present.

- [ ] **Step 1: Write failing fallback, local improvement, and replay tests**

```python
def make_depth_sequence():
    anchor = 1
    base = np.linspace(0.2, 0.8, 72 * 128, dtype=np.float32).reshape(72, 128)
    video = np.stack([base - 0.002, base, base + 0.002])
    photo = 0.15 + 0.8 * base
    return video, photo.astype(np.float32), anchor


def make_two_region_sequence():
    anchor = 1
    base = np.full((72, 128), 0.45, np.float32)
    video = np.stack([base, base, base])
    photo = base.copy()
    photo[:, :64] += 0.12
    photo[:, 64:] -= 0.08
    flow = DenseFlowSequence(
        to_next=np.zeros((2, 36, 64, 2), np.float32),
        to_previous=np.zeros((2, 36, 64, 2), np.float32),
        confidence_next=np.ones((2, 36, 64), np.float32),
        confidence_previous=np.ones((2, 36, 64), np.float32),
        scene_cuts=np.zeros(2, bool),
        frame_shape=(36, 64),
    )
    rgb = np.zeros((3, 72, 128, 3), np.uint8)
    rgb[:, :, 64:] = 255
    return video, photo, rgb, anchor, flow


def test_v5_without_flow_is_exact_v4_fallback():
    video, photo, anchor = make_depth_sequence()
    v4 = DepthSync(DepthSyncConfig(algorithm_version="v4")).offline_prepare(video, photo, anchor)
    v5 = DepthSync(DepthSyncConfig(algorithm_version="v5")).offline_prepare(video, photo, anchor)
    np.testing.assert_array_equal(v5.depths, v4.depths)
    assert v5.local_fields is None
    assert set(v5.fallback_reasons) == {"v5_flow_unavailable"}


def test_v5_local_anchor_improves_over_v4_and_replays():
    video, photo, rgb, anchor, flow = make_two_region_sequence()
    sync = DepthSync(DepthSyncConfig(algorithm_version="v5"))
    result = sync.offline_prepare(
        video, photo, anchor, rgb_frames=rgb, dense_flow=flow
    )
    v4 = DepthSync(DepthSyncConfig(algorithm_version="v4")).offline_prepare(video, photo, anchor)
    assert np.mean(np.abs(result.depths[anchor] - photo)) < np.mean(np.abs(v4.depths[anchor] - photo))
    replay = sync.apply_frame(video[anchor], result.parameters[anchor])
    np.testing.assert_allclose(replay, result.depths[anchor], atol=1e-6)
```

- [ ] **Step 2: Run integration tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_depthsync.py -k "v5" -v
```

Expected: fail because `dense_flow` and V5 result fields are unsupported.

- [ ] **Step 3: Add V5 configuration and result fields**

Add explicit configuration values:

```python
local_grid_shape=(36, 64)
local_scale_bounds=(0.5, 1.5)
local_offset_bound_fraction=0.35
local_min_confidence=1e-3
local_spatial_iterations=5
local_max_scale_step=0.03
local_max_offset_step_fraction=0.02
```

Keep all existing V4 defaults unchanged. Reject algorithm versions outside `{"v1", "v3", "v4", "v5"}`.

- [ ] **Step 4: Dispatch V5 after V4 base generation**

Refactor only enough to make the already computed V4 `base_outputs` reusable. When V5 and valid dense flow are present:

```python
anchor_field = fit_anchor_field(base_depths[anchor_index], photo, local_config)
local_fields = propagate_local_fields(
    base_depths,
    rgb_frames,
    anchor_field,
    anchor_index,
    dense_flow,
    local_config,
)
outputs = np.stack([
    apply_local_field(base, local_fields.frame(i), local_config)
    for i, base in enumerate(base_depths)
])
```

When flow is absent, invalid, or crosses an unhandled error boundary, return V4 depths and record a deterministic V5 fallback reason. Do not catch `KeyboardInterrupt` or process-termination exceptions.

- [ ] **Step 5: Run V5 and regression suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_depthsync.py tests/test_local_field.py tests/test_flow.py -v
```

Expected: all tests pass; V4 fixtures remain byte-for-byte stable.

- [ ] **Step 6: Commit integration**

```powershell
git add depthsync/core.py depthsync/__init__.py tests/test_depthsync.py
git commit -m "feat: integrate V5 local fields with V4 fallback"
```

---

### Task 6: Add the SEA-RAFT-S Preparation-Stage Adapter and Weight Manifest

**Files:**
- Create: `depthsync/sea_raft.py`
- Create: `tools/run_flow_model.py`
- Create: `tests/test_sea_raft.py`
- Modify: `config/model-sources.json`
- Modify: `README.md`

**Interfaces:**
- Produces `SeaRaftEstimator(checkpoint, repository, device, input_size=(144,256))`.
- `estimate_pairs(frames: Sequence[np.ndarray]) -> DenseFlowSequence`.
- `from_backend(backend, input_size)` accepts a test backend whose `predict(image1, image2)` returns `(flow[H,W,2], log_variance[H,W])`.
- CLI writes `artifacts/flow/<scene>/sea_raft_s_flow.npz` and `timing.json`.

- [ ] **Step 1: Write a failing adapter test with a fake backend**

```python
class FakeSeaRaftBackend:
    def __init__(self, flow, variance):
        self.flow = np.asarray(flow, np.float32)
        self.log_variance = float(variance)
        self.pair_count = 0

    def predict(self, image1, image2):
        self.pair_count += 1
        h, w = image1.shape[:2]
        flow = np.broadcast_to(self.flow, (h, w, 2)).copy()
        log_variance = np.full((h, w), self.log_variance, np.float32)
        return flow, log_variance


def test_adapter_batches_forward_and_reverse_pairs():
    backend = FakeSeaRaftBackend(flow=(2.0, 0.0), variance=0.0)
    estimator = SeaRaftEstimator.from_backend(backend, input_size=(16, 24))
    frames = [np.zeros((32, 48, 3), np.uint8) for _ in range(4)]
    result = estimator.estimate_pairs(frames)
    assert result.to_next.shape == (3, 16, 24, 2)
    assert result.to_previous.shape == (3, 16, 24, 2)
    assert backend.pair_count == 6
```

- [ ] **Step 2: Run the adapter test and confirm failure**

Run:

```powershell
.\.venv-models\Scripts\python.exe -m pytest tests/test_sea_raft.py -v
```

Expected: fail because `depthsync.sea_raft` does not exist.

- [ ] **Step 3: Implement lazy loading and uncertainty conversion**

Import PyTorch and the checked-out SEA-RAFT package only inside `SeaRaftEstimator.__init__`. Convert BGR OpenCV frames to RGB tensors, letterbox to a multiple of 8, infer adjacent pairs in forward and reverse order, remove padding, and resize flow to `256 × 144` with vector scale correction. Convert predicted log-variance to model confidence:

```python
model_confidence = np.exp(-0.5 * np.clip(log_variance, 0.0, 10.0))
```

Then call `compute_flow_confidence` for each direction. The module must remain importable in `.venv` without PyTorch.

- [ ] **Step 4: Pin model provenance**

Add a `sea_raft_small` entry to `config/model-sources.json` with:

```json
{
  "repository": "https://github.com/princeton-vl/SEA-RAFT",
  "revision": "9137517ba24e628442aec097d3afe71d03503b75",
  "config": "config/eval/spring-S.json",
  "huggingface_repo": "MemorySlices/Tartan-C-T-TSKH-spring540x960-S",
  "huggingface_revision": "31b9b4b711bd2d0d0d38cb99d93ba144beb2dc92",
  "checkpoint_file": "model.safetensors",
  "local_checkpoint": "models/sea_raft_s.safetensors",
  "sha256": "d6a75e47f2630ba6c354ce84a322e24c3d9def668a6956f340a054b9a3211908",
  "license": "BSD-3-Clause"
}
```

Keep `third_party/` and `models/` ignored; commit only provenance and setup commands. The README commands must clone the pinned revision, fetch the official checkpoint revision, verify the exact SHA-256 above, and run the CLI.

- [ ] **Step 5: Run unit and one-pair model smoke tests**

Run:

```powershell
.\.venv-models\Scripts\python.exe -m pytest tests/test_sea_raft.py -v
.\.venv-models\Scripts\python.exe tools/run_flow_model.py --scenes 01 --max-pairs 1 --device cuda
```

Expected: unit tests pass; smoke output contains finite forward/backward flow, nonzero confidence, model revision, checkpoint hash, measured milliseconds, peak process RSS, and peak GPU allocated bytes when CUDA is used.

- [ ] **Step 6: Generate cached flow for all 90-frame clips**

Run:

```powershell
.\.venv-models\Scripts\python.exe tools/run_flow_model.py --scenes 01 02 03 --device cuda
```

Expected: each scene has 89 forward pairs, 89 backward pairs, no shape mismatch, and a timing file under `artifacts/flow/<scene>/`.

- [ ] **Step 7: Commit the adapter and reproducibility metadata**

```powershell
git add depthsync/sea_raft.py tools/run_flow_model.py tests/test_sea_raft.py config/model-sources.json README.md
git commit -m "feat: add SEA-RAFT-S preparation adapter"
```

---

### Task 7: Add V5 Metrics and Automated Acceptance Gates

**Files:**
- Create: `tests/test_evaluate.py`
- Modify: `depthsync/evaluate.py`
- Modify: `config/validation-scenes.json`

**Interfaces:**
- Produces `evaluate_v5_gates(scene, raw, v4, v5, photo, flow, scene_config) -> dict`.
- Adds `--flow-root` and `--algorithm-version` CLI arguments.
- Saves `v4_depth.npz`, `v5_depth.npz`, `v5_parameters.npz`, and `metrics.json`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_flat_region_correction_gradient_ignores_real_depth_edges():
    base = np.full((16, 16), 0.3, np.float32)
    base[:, 8:] = 0.8
    synced = base + 0.01
    synced[:, 8:] += 0.5
    value = flat_correction_gradient_p99(synced, base, flat_threshold=0.02)
    assert value < 1e-6


def test_gate_rejects_temporal_regression():
    metrics = {
        "v4_subject_temporal_p95": 0.020,
        "v5_subject_temporal_p95": 0.020,
        "v4_subject_anchor_nmae": 0.10,
        "v5_subject_anchor_nmae": 0.07,
        "v4_wall_flat_correction_gradient_p99": 0.01,
        "v5_wall_flat_correction_gradient_p99": 0.01,
        "v5_revealed_background_confidence_p95": 0.02,
        "v4_excess_jump_max": 0.001,
        "v5_excess_jump_max": 0.001,
        "v5_parameter_bytes": 1_244_160,
    }
    metrics["v5_subject_temporal_p95"] = 1.051 * metrics["v4_subject_temporal_p95"]
    gates = classify_v5_gates("02", metrics)
    assert gates["02_subject_temporal"]["passed"] is False
```

- [ ] **Step 2: Run metric tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -v
```

Expected: fail because V5 metric helpers do not exist.

- [ ] **Step 3: Implement exact gates**

Add these machine-readable gates:

```text
01_subject_anchor_nmae: V5 <= 0.80 * V4
01_wall_flat_correction_gradient_p99: V5 <= V4 + 1e-6
02_subject_temporal_p95: V5 <= 1.05 * V4
02_revealed_background_confidence_p95: <= 0.05
03_excess_jump_max: V5 <= V4 + 2e-4
all_parameter_bytes: <= 1,244,160 for persisted FP16 local channels
```

Use flow-compensated temporal error for subject and global metrics. Add exact scene configuration entries for `01` wall/subject crops, `02` tail subject/revealed-background crops, and `03` subject crop. Frame-specific visual gates remain explicitly marked `manual_pass: null` until inspected; the report cannot claim overall pass while any manual gate is null.

- [ ] **Step 4: Run metric tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py tests/test_depthsync.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit metrics**

```powershell
git add depthsync/evaluate.py config/validation-scenes.json tests/test_evaluate.py
git commit -m "feat: add V5 regional and temporal gates"
```

---

### Task 8: Generate Fixed-Range Depth and Field Diagnostic Videos

**Files:**
- Modify: `depthsync/depth_visualization.py`
- Create: `tests/test_depth_visualization.py`

**Interfaces:**
- Reads V4/V5 depth and V5 parameter artifacts.
- `render_scene_from_arrays(data: dict[str, np.ndarray | float | int], output_dir, write_video=True) -> dict`.
- Writes `depth_comparison_v5.mp4`, `field_diagnostics_v5.mp4`, and designated PNGs.

- [ ] **Step 1: Write failing fixed-range and artifact-layout tests**

```python
def test_v5_visualization_uses_one_photo_range_for_all_depth_panels(tmp_path):
    depth = np.linspace(0.1, 0.9, 3 * 12 * 16, dtype=np.float32).reshape(3, 12, 16)
    fields = np.zeros((3, 6, 8), np.float32)
    fixture = {
        "raw": depth,
        "v4": depth + 0.01,
        "v5": depth + 0.02,
        "photo": depth[1] + 0.03,
        "delta_scale": fields,
        "offset_norm": fields,
        "confidence": fields,
        "flow_consistency": np.ones_like(fields),
        "fps": 30.0,
        "anchor_index": 1,
    }
    metadata = render_scene_from_arrays(
        fixture, tmp_path, write_video=False
    )
    assert metadata["display_range_mode"] == "photo_anchor_p02_p98"
    assert metadata["depth_panel_limits"]["v4"] == metadata["depth_panel_limits"]["v5"]
    assert metadata["depth_panel_limits"]["v5"] == metadata["depth_panel_limits"]["photo"]


def test_required_inspection_frames_are_declared():
    assert required_frames("01") == [0, 45, 60]
    assert required_frames("02") == list(range(75, 90))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_depth_visualization.py -v
```

Expected: fail because V5 visualization helpers are absent.

- [ ] **Step 3: Implement four-panel outputs**

Depth comparison layout:

```text
VDA raw | V4 global LUT | V5 local field | DepthPro anchor
```

Diagnostic layout:

```text
delta_scale | offset_norm | confidence | flow consistency
```

Use the photo anchor P02–P98 range for V4, V5, and DepthPro panels across every video frame. Use fixed `[-0.5,0.5]`, `[-0.35,0.35]`, `[0,1]`, and `[0,1]` ranges for the four diagnostic panels. Export full-resolution PNGs for 01 frames 0/45/60, 02 frames 75–89, and 03’s maximum-jump frame plus adjacent frames.

- [ ] **Step 4: Run visualization tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_depth_visualization.py -v
```

Expected: all tests pass and the in-memory rendering path does not require FFmpeg.

- [ ] **Step 5: Commit visualization**

```powershell
git add depthsync/depth_visualization.py tests/test_depth_visualization.py
git commit -m "feat: visualize V5 depth and local fields"
```

---

### Task 9: Run the Three-Scene Validation and Tune Only Declared V5 Parameters

**Files:**
- Create: `reports/validation-v5.md`
- Modify only when evidence requires it: `depthsync/local_field.py`
- Modify only when evidence requires it: `depthsync/flow.py`
- Modify only when evidence requires it: `config/validation-scenes.json`
- Modify corresponding tests for every algorithm change.

**Interfaces:**
- Consumes cached depth under `artifacts/depth/{01,02,03}` and cached flow under `artifacts/flow/{01,02,03}`.
- Produces final metrics and videos under `results/v5/{01,02,03}`.

- [ ] **Step 1: Establish the untouched V4 baseline**

Run:

```powershell
.\.venv\Scripts\python.exe -m depthsync.evaluate --scenes 01 02 03 --algorithm-version v4 --result-root results/v5-baseline
```

Expected: metrics reproduce commit `e5c58e1` within `1e-6` for deterministic metrics.

- [ ] **Step 2: Run V5 and generate videos**

Run:

```powershell
.\.venv\Scripts\python.exe -m depthsync.evaluate --scenes 01 02 03 --algorithm-version v5 --flow-root artifacts/flow --result-root results/v5
.\.venv\Scripts\python.exe -m depthsync.depth_visualization --scenes 01 02 03 --result-root results/v5
```

Expected: all automatic gates are present, no metric is NaN, and every requested MP4/PNG exists.

- [ ] **Step 3: Inspect the specified frames before tuning averages**

Inspect:

```text
results/v5/01/depth_visualization/frame_000.png
results/v5/01/depth_visualization/frame_060.png
results/v5/02/depth_visualization/frame_075.png through frame_089.png
results/v5/03/depth_visualization/worst_jump_previous.png
results/v5/03/depth_visualization/worst_jump.png
results/v5/03/depth_visualization/worst_jump_next.png
```

Record each manual gate as pass/fail with a one-sentence observation in `reports/validation-v5.md`. A failed specified-frame gate blocks acceptance even if average metrics improve.

- [ ] **Step 4: Diagnose each failed gate from saved fields**

Use `field_diagnostics_v5.mp4` and per-frame arrays to classify failure into exactly one first cause:

```text
anchor fit error
flow inconsistency not rejected
confidence leakage into revealed region
spatial regularization crossing a depth edge
trajectory filter lag
visualization-only quantization
```

Before changing code, add a synthetic regression test reproducing the cause. Change one declared V5 parameter group at a time: fit window/robustness, flow confidence, spatial weights, or temporal weights. Do not change V4.

- [ ] **Step 5: Re-run all three scenes after every accepted tuning change**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m depthsync.evaluate --scenes 01 02 03 --algorithm-version v5 --flow-root artifacts/flow --result-root results/v5
.\.venv\Scripts\python.exe -m depthsync.depth_visualization --scenes 01 02 03 --result-root results/v5
```

Expected: tests pass, all automatic gates pass, and no previously passed manual frame regresses.

- [ ] **Step 6: Commit validated tuning**

```powershell
git add depthsync/local_field.py depthsync/flow.py tests config/validation-scenes.json reports/validation-v5.md
git commit -m "fix: tune V5 local alignment on validation scenes"
```

If defaults pass without changes, commit only `reports/validation-v5.md` with message `docs: record V5 validation results`.

---

### Task 10: Update the Algorithm Documentation and Draw.io Flow

**Files:**
- Modify: `README.md`
- Modify: `docs/depthsync-flow.drawio`
- Modify: `docs/depthsync-flow.png`
- Create: `docs/depthsync-v5-algorithm.md`

**Interfaces:**
- Documents the verified implementation and measured data, not the pre-implementation target alone.

- [ ] **Step 1: Update detailed algorithm documentation**

Document exact inputs, normalized disparity convention, tensor layouts, FP16 persisted field format, temporary flow buffers, preparation/playback sequence, failure fallback, measured timing, memory, and commands. Include this byte calculation:

```text
90 frames × 64 × 36 grid × 3 channels × 2 bytes = 1,244,160 bytes
```

- [ ] **Step 2: Replace the Draw.io V4-only flow with V5 preparation and playback swimlanes**

The source diagram must show:

```text
Preparation:
RGB pairs -> SEA-RAFT-S bidirectional flow -> consistency confidence
V4 base + photo anchor -> local affine fit
fit + flow -> propagate -> regularize/filter -> FP16 field package

Playback:
video depth -> V4 LUT -> depth-guided field upsample -> confidence blend -> V5 depth
failure/zero confidence -----------------------------------------> V4 depth
```

Keep `.drawio` editable and export the matching PNG.

- [ ] **Step 3: Verify documentation against actual artifacts**

Run:

```powershell
rg -n "1,244,160|64 × 36|SEA-RAFT|confidence|V4" README.md docs/depthsync-v5-algorithm.md
git diff --check
```

Expected: every required term is present and the diff has no whitespace errors.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md docs/depthsync-v5-algorithm.md docs/depthsync-flow.drawio docs/depthsync-flow.png
git commit -m "docs: document validated DepthSync V5 pipeline"
```

---

### Task 11: Final Verification, Review, and Remote Publication

**Files:**
- No intended source changes.

**Interfaces:**
- Produces a clean, reviewed branch with reproducible local result paths.

- [ ] **Step 1: Run the full automated suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Re-run all validation and visualization commands from a clean process**

Run:

```powershell
.\.venv\Scripts\python.exe -m depthsync.evaluate --scenes 01 02 03 --algorithm-version v5 --flow-root artifacts/flow --result-root results/v5
.\.venv\Scripts\python.exe -m depthsync.depth_visualization --scenes 01 02 03 --result-root results/v5
```

Expected: automatic gate summary passes, manual gates in `reports/validation-v5.md` are all resolved, and all videos open with 90 frames at 30 fps.

- [ ] **Step 3: Verify Git and artifact boundaries**

Run:

```powershell
git status --short
git diff --check
git ls-files models third_party artifacts results
git log --oneline origin/codex/edge-lite-depthsync..HEAD
```

Expected: worktree is clean; generated media, model weights, cloned third-party code, and cached arrays are not tracked; the log shows task-sized commits.

- [ ] **Step 4: Perform completion review**

Use `superpowers:requesting-code-review`, address findings with `superpowers:receiving-code-review`, then use `superpowers:verification-before-completion`. Do not claim V5 is better unless both the metrics and specified-frame inspection support it.

- [ ] **Step 5: Push the V5 branch**

Run:

```powershell
git push -u origin codex/v5-flow-local-affine
```

Expected: branch is available at `luckystone60/DepthSync` and the local branch tracks the remote.
