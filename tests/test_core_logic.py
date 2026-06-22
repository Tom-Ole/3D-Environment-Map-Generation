"""Lightweight logic tests for the parts of the pipeline that depend only on
numpy.

These run without the heavy optional dependencies (torch, open3d, opencv,
PySide6, bosdyn) so they can be executed in any environment:

    python3 tests/test_core_logic.py

They cover the behaviour that was changed during the audit:
  - keyframe interval selection and the max-frames cap
  - motion-based keyframe selection resetting per camera source
  - pipeline overall-progress arithmetic
  - geometric model intrinsics scaling with image resize
  - the shared to_numpy helper
"""

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from ai_reconstruction.keyframe import select_keyframes  # noqa: E402
from ai_reconstruction.models.base import (  # noqa: E402
    pairs_budget_for_memory,
    resolve_scene_graph,
    to_numpy,
)
from ai_reconstruction.models.geometric_model import GeometricModel  # noqa: E402
from ai_reconstruction.pipeline import AIReconstructionPipeline  # noqa: E402
from ai_reconstruction.types import (  # noqa: E402
    AIPointCloudResult,
    AIReconstructionConfig,
    ImageRecord,
    KeyframeStrategy,
)

_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def _records(source: str, n: int, ts0: float = 0.0) -> list:
    return [
        ImageRecord(path=Path(f"{i:05d}_{source}.png"), source_name=source,
                    frame_id=i, inferred_timestamp=ts0)
        for i in range(n)
    ]


def test_keyframe_interval():
    recs = _records("frontleft_fisheye_image", 20)
    out = select_keyframes(recs, strategy=KeyframeStrategy.INTERVAL,
                           interval=5, max_frames=100)
    check("interval selects every Nth frame", len(out) == 4,
          f"expected 4 got {len(out)}")
    check("interval keeps first frame", out[0].frame_id == 0)


def test_keyframe_max_cap():
    recs = _records("frontleft_fisheye_image", 200)
    out = select_keyframes(recs, strategy=KeyframeStrategy.INTERVAL,
                           interval=1, max_frames=50)
    check("max_frames cap enforced", len(out) == 50,
          f"expected 50 got {len(out)}")


def test_motion_resets_per_source():
    # One pose at t=100 at the origin; record timestamps (0.0) fall outside the
    # pose time range, so selection uses the nearest pose (no scipy needed) and
    # every record maps to the same position -> zero motion within a source.
    poses = np.array([[100.0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)
    recs = _records("frontleft_fisheye_image", 3) + _records("frontright_fisheye_image", 3)
    out = select_keyframes(recs, strategy=KeyframeStrategy.MOTION,
                           spot_poses=poses, min_translation=0.3,
                           min_rotation_deg=10.0, max_frames=100)
    sources = {r.source_name for r in out}
    check("motion selection keeps the first frame of each source", len(out) == 2,
          f"expected 2 got {len(out)}")
    check("motion selection covers both cameras", sources == {
        "frontleft_fisheye_image", "frontright_fisheye_image"})


def test_emit_overall_pct():
    captured = {}
    pipe = AIReconstructionPipeline(
        session_path=Path("/tmp/nonexistent_session"),
        config=AIReconstructionConfig(),
        progress_callback=lambda p: captured.update(overall=p.overall_pct),
    )
    # Stage 3 of 6 at 50% -> ((3-1) + 0.5) / 6 * 100 = 41.666...
    pipe._emit("inference", 3, 50.0, "halfway")
    check("overall progress arithmetic", abs(captured["overall"] - 41.6667) < 1e-3,
          f"got {captured.get('overall')}")


def test_geometric_intrinsics_scale():
    intr = {"frontleft_fisheye_image": {"fx": 500, "fy": 500, "cx": 320, "cy": 240}}
    model = GeometricModel(device="cpu", intrinsics=intr)
    path = Path("00001_frontleft_fisheye_image.png")
    k_full = model._get_K(path, scale=1.0)
    k_half = model._get_K(path, scale=0.5)
    check("intrinsics scale with resize",
          np.allclose(k_half[:2, :3], k_full[:2, :3] * 0.5),
          f"\n{k_half}")
    check("intrinsics homogeneous row preserved",
          np.allclose(k_half[2], [0, 0, 1]))


def test_pairs_budget_for_memory():
    # Higher resolution -> fewer pairs fit the same budget (quadratic in size).
    check("budget shrinks with resolution",
          pairs_budget_for_memory(512, 8.0) > pairs_budget_for_memory(1024, 8.0))
    # More memory -> more pairs.
    check("budget grows with memory",
          pairs_budget_for_memory(512, 16.0) > pairs_budget_for_memory(512, 8.0))
    check("budget is always positive", pairs_budget_for_memory(512, 0.1) >= 1)


def test_resolve_scene_graph():
    # Small session keeps the complete graph.
    graph, down = resolve_scene_graph("auto", 5, max_pairs=1000)
    check("small session uses complete", graph == "complete" and not down)

    # Large session is downgraded to a window that fits the budget.
    graph, down = resolve_scene_graph("auto", 100, max_pairs=1000)
    check("large session downgraded to window", graph.startswith("swin-") and down,
          f"got {graph}")
    window = int(graph.split("-")[1])
    check("window fits the pair budget", 2 * window * 100 <= 1000,
          f"window={window}")

    # Explicit graphs are honoured unchanged.
    graph, down = resolve_scene_graph("swin-3", 100, max_pairs=10)
    check("explicit graph honoured", graph == "swin-3" and not down)

    # 'complete' is protected from blowups too (memory is a hard limit).
    graph, down = resolve_scene_graph("complete", 100, max_pairs=1000)
    check("complete downgraded when over budget", graph.startswith("swin-") and down,
          f"got {graph}")


def test_postprocess_drops_nonfinite():
    # NaN/Inf coordinates must be removed before any Open3D call (they crash
    # Open3D natively). Small metric cloud -> only the finite filter runs.
    pts = np.array([[0, 0, 1], [1, 1, 1], [np.nan, 0, 1],
                    [np.inf, 0, 1], [2, 2, 2]], np.float32)
    raw = AIPointCloudResult(points=pts, colors=np.zeros((5, 3), np.uint8),
                             metric_scale=True)
    pipe = AIReconstructionPipeline(Path("/tmp/x"), AIReconstructionConfig())
    out = pipe._postprocess(raw)
    check("non-finite points removed", len(out.points) == 3,
          f"got {len(out.points)}")
    check("no NaN/Inf remain", bool(np.isfinite(out.points).all()))


def test_postprocess_clips_extreme_outliers():
    # A normal cluster plus one huge-but-finite flyer; the flyer must be
    # clipped before Open3D so it cannot blow up the voxel grid extent.
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.0, 1.0, size=(40, 3)).astype(np.float32)
    flyer = np.array([[1e9, 1e9, 1e9]], np.float32)
    pts = np.vstack([cluster, flyer])
    raw = AIPointCloudResult(points=pts, colors=np.zeros((41, 3), np.uint8),
                             metric_scale=True)
    pipe = AIReconstructionPipeline(Path("/tmp/x"), AIReconstructionConfig())
    out = pipe._postprocess(raw)
    check("extreme flyer clipped", len(out.points) == 40,
          f"got {len(out.points)}")
    check("cluster retained, extent bounded",
          float(np.abs(out.points).max()) < 1e6)


def test_to_numpy_passthrough():
    arr = np.arange(6).reshape(2, 3)
    out = to_numpy(arr)
    check("to_numpy returns ndarray", isinstance(out, np.ndarray))
    check("to_numpy preserves values", np.array_equal(out, arr))


def main():
    tests = [
        test_keyframe_interval,
        test_keyframe_max_cap,
        test_motion_resets_per_source,
        test_emit_overall_pct,
        test_geometric_intrinsics_scale,
        test_pairs_budget_for_memory,
        test_resolve_scene_graph,
        test_postprocess_drops_nonfinite,
        test_postprocess_clips_extreme_outliers,
        test_to_numpy_passthrough,
    ]
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print(f"\n{'=' * 40}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
