"""Headless runner for the AI reconstruction pipeline.

Run this in a normal terminal (NOT the VSCode integrated terminal) so that an
out-of-memory kill is visible and the per-stage RSS log is not swallowed:

    python3 run_ai_reconstruction.py recordings/notebook/20260610_153903 \
        --model geometric --max-images 40 --image-size 384

Each pipeline stage logs its resident memory as "RSS=____MB". If the process
is killed, the LAST RSS line printed identifies the stage that ran out of
memory. Share that output to pinpoint the bottleneck.

Recommended low-memory first run (cannot OOM): --model geometric
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Make the locally cloned MASt3R / DUSt3R importable, exactly as main.py does.
_mast3r_repo = Path(__file__).parent / "mast3r"
if _mast3r_repo.is_dir() and (_mast3r_repo / "mast3r" / "__init__.py").is_file():
    sys.path.insert(0, str(_mast3r_repo))
    _dust3r_repo = _mast3r_repo / "dust3r"
    if _dust3r_repo.is_dir() and (_dust3r_repo / "dust3r" / "__init__.py").is_file():
        sys.path.insert(0, str(_dust3r_repo))

from ai_reconstruction.pipeline import AIReconstructionPipeline
from ai_reconstruction.types import (
    AIReconstructionConfig,
    DeviceType,
    KeyframeStrategy,
    ModelType,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless AI reconstruction runner")
    p.add_argument("session", type=str, help="Path to a recorded session folder")
    p.add_argument("--model", default="auto",
                   choices=[m.value for m in ModelType])
    p.add_argument("--device", default="auto",
                   choices=[d.value for d in DeviceType])
    p.add_argument("--max-images", type=int, default=40,
                   help="Keyframe cap (lower this if memory is tight)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--keyframe-strategy", default="interval",
                   choices=[k.value for k in KeyframeStrategy])
    p.add_argument("--keyframe-interval", type=int, default=5)
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--max-pair-memory-gb", type=float, default=8.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    session = Path(args.session)
    if not session.exists():
        print(f"Session not found: {session}")
        return 2

    config = AIReconstructionConfig(
        model_type=ModelType(args.model),
        device=DeviceType(args.device),
        image_size=args.image_size,
        max_images=args.max_images,
        keyframe_strategy=KeyframeStrategy(args.keyframe_strategy),
        keyframe_interval=args.keyframe_interval,
        voxel_size=args.voxel_size,
        max_pair_memory_gb=args.max_pair_memory_gb,
    )

    print(f"Running AI reconstruction: model={args.model} device={args.device} "
          f"max_images={args.max_images} image_size={args.image_size}")
    pipeline = AIReconstructionPipeline(session_path=session, config=config)
    result = pipeline.run()

    if result.success:
        print(f"\nSUCCESS: {result.point_count:,} points in "
              f"{result.duration_seconds:.1f}s -> {result.point_cloud_path}")
        return 0
    print(f"\nFAILED: {result.error_message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
