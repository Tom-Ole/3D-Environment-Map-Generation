# AI Reconstruction Pipeline — Architecture & Design

## 1. Overview

The AI Reconstruction Pipeline is a fully independent camera-based 3D
reconstruction system that runs alongside the existing LiDAR-SLAM pipeline.
It consumes image data recorded by the existing SPOT capture system and
produces a dense coloured point cloud using state-of-the-art AI models.

```
Recording session  (existing)
        |
        v
  images/*.png        poses.npy       intrinsics.json
        |                  |                |
        +------------------+----------------+
                           |
                    AI Reconstruction
                    Pipeline (new)
                           |
            +--------------+--------------+
            |              |              |
     point_cloud.ply  camera_poses.npy  metadata.json
            (ai_reconstruction/ subfolder)
```

---

## 2. Research & Model Selection

### Models Evaluated

| Model | Year | Type | Scale | Speed | Notes |
|---|---|---|---|---|---|
| **MASt3R** | ECCV 2024 | Dense + matching | **Metric** | Moderate | Best for robot rooms |
| **DUSt3R** | CVPR 2024 | Dense pairs | Arbitrary | Slower | Foundation of MASt3R |
| **VGGT** | CVPR 2025 | Single-pass | Arbitrary | Very fast | Good for quick previews |
| **Geometric** | N/A | ORB + triangulation | **Metric** | Very fast | CPU, always available |

### Selection Rationale

**Primary: MASt3R**

MASt3R was selected as the default AI model for the following reasons:

1. **Metric scale output** — the model is trained with metric depth supervision,
   so the reconstructed cloud is in real-world metres without needing a
   scale-fixing reference frame.  This is essential for integration with the
   LiDAR-SLAM outputs.

2. **Sparse global alignment** — unlike DUSt3R which runs a dense iterative
   optimizer, MASt3R uses explicit feature matching heads to build a sparse
   correspondence graph, making it 3–10× faster on large scenes (> 30 frames).

3. **Indoor robustness** — trained on diverse indoor datasets, it handles the
   low-texture walls and reflective floors common in SPOT deployments.

**Secondary: DUSt3R**

DUSt3R is kept as the secondary option because its dense point-map output
provides better coverage in areas with sparse ORB features (smooth floors,
blank walls).

**Tertiary: VGGT**

VGGT's single forward-pass design makes it the fastest option (~5× faster
than MASt3R on a modern GPU).  It is suitable for quick previews or when
compute is constrained.  Accuracy on complex indoor scenes is lower.

**Always-available fallback: Geometric**

The Geometric model requires only `opencv-python` (already a project
dependency) and no model download.  It exploits SPOT's highly accurate VIO
poses to skip pose estimation entirely, triangulating ORB matches directly.
This is the recommended choice when:
- No GPU is available
- Internet access is restricted (no model download possible)
- A quick sanity check is needed

---

## 3. Architecture

### Module Layout

```
src/
├── ai_reconstruction/              NEW — completely independent from LiDAR-SLAM
│   ├── __init__.py
│   ├── types.py                    Dataclasses: config, progress, results
│   ├── image_loader.py             Discover & load session images
│   ├── keyframe.py                 Interval / motion-based selection
│   ├── export.py                   Write PLY + poses + metadata.json
│   ├── pipeline.py                 6-stage orchestrator
│   └── models/
│       ├── __init__.py             Registry: get_model(), get_available_models()
│       ├── base.py                 Abstract ReconstructionModel
│       ├── mast3r_model.py         MASt3R wrapper
│       ├── dust3r_model.py         DUSt3R wrapper
│       ├── vggt_model.py           VGGT wrapper
│       └── geometric_model.py      CPU ORB-triangulation fallback
│
├── gui/
│   ├── tabs/
│   │   ├── reconstruct_tab.py      UNCHANGED (LiDAR-SLAM, now labelled)
│   │   └── ai_reconstruct_tab.py   NEW — AI pipeline GUI
│   └── workers/
│       ├── reconstruct_worker.py   UNCHANGED (LiDAR-SLAM)
│       └── ai_reconstruct_worker.py  NEW — AI pipeline QThread worker
│
└── config.py                       Extended with ai_* fields (non-breaking)
```

### Separation From LiDAR-SLAM

| Concern | LiDAR-SLAM | AI Pipeline |
|---|---|---|
| Input | `lidar/*.ply` scans | `images/*.png` photos |
| Python module | `src/reconstruction/` | `src/ai_reconstruction/` |
| GUI tab | "LiDAR-SLAM" | "AI Reconstruction" |
| Worker | `reconstruct_worker.py` | `ai_reconstruct_worker.py` |
| Output folder | `session/reconstruction/` | `session/ai_reconstruction/` |
| Config prefix | `voxel_size`, `icp_*` | `ai_*` |

Neither pipeline imports from the other.  They share only the recording
session helpers (`recording/session.py`) for loading common assets.

---

## 4. Pipeline Stages

```
Stage 1  load_session
         list_session_images()  →  ImageRecord list
         load_camera_intrinsics()
         load_poses()  (optional, for geometric + keyframe motion)

Stage 2  select_keyframes
         INTERVAL: every Nth image
         MOTION:   new frame when Δtranslation ≥ threshold OR Δrotation ≥ threshold

Stage 3  load_model
         get_model(type, device, **kwargs)  →  ReconstructionModel
         model.load()  (downloads weights on first run)

Stage 4  inference
         model.reconstruct(image_paths, progress_cb)  →  AIPointCloudResult

Stage 5  postprocess
         confidence filter (DUSt3R / MASt3R only)
         depth range clamp (skip if already metric XYZ)
         voxel downsample (Open3D)
         statistical outlier removal (Open3D)

Stage 6  export
         point_cloud.ply  (Open3D binary PLY, or ASCII fallback)
         camera_poses.npy  (Mx4x4 float64)
         metadata.json  (model, params, timing)
```

### Progress Reporting

Every stage emits `AIReconstructionProgress` objects carrying:
- `stage_pct`   — completion within the current stage (0–100)
- `overall_pct` — completion across all 6 stages (0–100)

The GUI tab displays both via two stacked `QProgressBar` widgets.

---

## 5. Model Interface

Every model implements three methods:

```python
class ReconstructionModel(ABC):
    @classmethod
    def is_available(cls) -> bool: ...     # imports probed at import time

    def load(self) -> None: ...            # download / GPU-load weights

    def reconstruct(
        self,
        image_paths: List[Path],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> AIPointCloudResult: ...
```

`AIPointCloudResult` carries:

| Field | Type | Description |
|---|---|---|
| `points` | `Nx3 float32` | 3-D world coordinates |
| `colors` | `Nx3 uint8` | RGB from source images |
| `confidence` | `N float32` | per-point quality (model-specific) |
| `camera_poses` | `Mx4x4 float64` | camera-to-world (when available) |
| `metric_scale` | `bool` | True if output is in real metres |

### Adding a New Model

1. Create `src/ai_reconstruction/models/my_model.py` inheriting `ReconstructionModel`.
2. Implement the three abstract methods.
3. Register in `src/ai_reconstruction/models/__init__.py`:
   - Add to `_AUTO_PRIORITY` and `registry` dict in `get_model()`.
   - Add to `get_available_models()`.
4. Add a GUI option in `ai_reconstruct_tab.py::_MODEL_OPTIONS`.

No changes to the pipeline, worker, or other models are required.

---

## 6. Data Flow

### Image Loading

```
session/images/00042_frontleft_fisheye_image.png
                 |              |
              frame_id     source_name
```

`list_session_images()` parses filenames, groups by source, and infers
timestamps as `session_start + source_index × 0.2 s`.  This is an
approximation sufficient for pose matching.

### Pose Matching (Geometric Model)

The Geometric model matches each keyframe to the nearest SPOT pose by
timestamp, then inverts the body pose to obtain the world-to-camera
extrinsic `[R_cw | t_cw]` used in triangulation.

No camera-to-body extrinsic is applied (assumed coincident).  This
introduces a small systematic offset (< 30 cm for typical SPOT mounting).

### Keyframe Motion Strategy

```python
# Select keyframe when EITHER condition triggers:
delta_t = ||pos_current - pos_last||     >= min_translation  (default: 0.3 m)
delta_r = 2 * arccos(|q_current · q_last|)  >= min_rotation  (default: 10°)
```

---

## 7. Dependencies

### Required (already in requirements.txt)

| Package | Use |
|---|---|
| `opencv-python` | Image loading, ORB, triangulation (Geometric model) |
| `numpy` | All numerical operations |
| `scipy` | SLERP for pose interpolation in keyframe selection |
| `open3d` | Post-processing, PLY export |

### Optional (install per model)

| Package | Model | Install |
|---|---|---|
| `torch` | MASt3R, DUSt3R, VGGT | `pip install torch` |
| `mast3r` | MASt3R | `pip install git+https://github.com/naver/mast3r.git` |
| `dust3r` | DUSt3R (also needed by MASt3R) | `pip install git+https://github.com/naver/dust3r.git` |
| `vggt` | VGGT | `pip install git+https://github.com/facebookresearch/vggt.git` |

The pipeline detects installed packages at startup via `is_available()` and
shows the availability status in the GUI tab's right panel.

---

## 8. Configuration

AI pipeline parameters are stored in `Config` with the `ai_` prefix:

| Field | Default | Description |
|---|---|---|
| `ai_model` | `"auto"` | Model selection |
| `ai_device` | `"auto"` | Compute device |
| `ai_image_size` | `512` | Resize long edge (px) |
| `ai_max_images` | `100` | Keyframe hard cap |
| `ai_keyframe_interval` | `5` | Every Nth frame (interval mode) |
| `ai_voxel_size` | `0.05` | Downsample voxel (m) |
| `ai_confidence_threshold` | `1.5` | DUSt3R/MASt3R confidence mask |

The GUI tab exposes all these as form controls and overrides the config
values per-run.

---

## 9. Output Format

```
session/ai_reconstruction/
├── point_cloud.ply     Binary PLY, properties: x y z red green blue
├── camera_poses.npy    NumPy array, shape (M, 4, 4), dtype float64
│                       Camera-to-world homogeneous transforms
└── metadata.json       {
                          "model": "mast3r",
                          "metric_scale": true,
                          "point_count": 142857,
                          "keyframe_count": 48,
                          "image_count": 48,
                          "device": "cuda",
                          "image_size": 512,
                          "camera_sources": ["frontleft_fisheye_image", ...],
                          "duration_seconds": 87.3,
                          "completed_at": "2026-06-15T13:42:01"
                        }
```

---

## 10. Limitations & Known Issues

1. **No fisheye undistortion** — DUSt3R/MASt3R tolerate moderate fisheye
   distortion but accuracy degrades at the frame periphery.  For best results,
   undistort images using the Kannala-Brandt coefficients in `intrinsics.json`
   before running AI models.  The Geometric model is not affected (it uses the
   stored focal length directly).

2. **Image timestamp approximation** — per-image timestamps are inferred from
   session start time + frame index × 0.2 s.  Drift up to ±0.5 s is possible.
   This affects keyframe motion selection and Geometric model pose matching.

3. **No loop closure in AI pipeline** — the AI models perform global alignment
   internally but do not run an explicit loop-closure step.  For very long
   trajectories (> 5 min), drift may accumulate.

4. **VRAM scaling** — DUSt3R runs `O(N²)` pairs for a complete scene graph.
   100 keyframes = 9,900 pairs.  Use `scene_graph="swin-5"` or reduce
   `max_images` if you run out of VRAM.

5. **CPU inference** — MASt3R/DUSt3R on CPU take ~2–10 minutes per pair.
   For CPU-only machines, the Geometric model is strongly recommended.

---

## 11. Extension Points

| What to extend | Where |
|---|---|
| Add a new AI model | `models/my_model.py` + register in `models/__init__.py` |
| Add a new keyframe strategy | `keyframe.py::select_keyframes` |
| Change output format | `export.py::export_results` |
| Add mesh generation | New stage in `pipeline.py` after postprocess |
| Add GUI visualisation | `ai_reconstruct_tab.py` right panel |
| Integrate with LiDAR cloud | Post-processing step reading both output folders |

---

## 12. Future Improvements

- **Fisheye undistortion**: apply `cv2.fisheye.undistortImage` before model
  inference using the Kannala-Brandt coefficients already stored in
  `intrinsics.json`.
- **Colour-mapped LiDAR fusion**: project fused LiDAR cloud into camera images
  to colour-transfer using the accurate geometry from the LiDAR-SLAM pipeline.
- **SLAM integration**: feed AI-estimated camera poses back into the pose-graph
  optimiser to serve as additional constraints between LiDAR scans.
- **Per-image timestamp sidecars**: write `images/{frame_id}.json` in the
  recording writer (mirrors the LiDAR sidecar design) for accurate timestamp
  matching.
- **Streaming mode**: process frames on-the-fly during recording for real-time
  3D previews via the capture tab.
