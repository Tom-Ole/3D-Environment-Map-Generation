"""
CLI: project camera images onto a LiDAR-SLAM mesh to produce a vertex-coloured PLY.

Usage
-----
    python colorize_mesh.py <session_path> [options]

    python colorize_mesh.py recordings/20240315_143022
    python colorize_mesh.py recordings/20240315_143022 --cameras frontleft_fisheye_image frontright_fisheye_image
    python colorize_mesh.py recordings/20240315_143022 --mesh path/to/custom.ply --out out.ply

Prerequisites
-------------
The reconstruction pipeline must have been run at least once so that
  session/reconstruction/mesh.ply  (or mesh.obj)
  session/reconstruction/keyframe_poses.npy
  session/reconstruction/keyframe_frame_ids.npy
all exist.  If keyframe_poses.npy is missing, re-run the pipeline with the
updated pipeline.py (which saves these files automatically).
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Ensure src/ is on the Python path when running from the project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from reconstruction.colorize import colorize_mesh  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Colorize a LiDAR-SLAM mesh using recorded camera images."
    )
    p.add_argument(
        "session",
        metavar="SESSION_PATH",
        type=Path,
        help="Path to the recorded session folder (e.g. recordings/20240315_143022).",
    )
    p.add_argument(
        "--mesh",
        metavar="PATH",
        type=Path,
        default=None,
        help="Override mesh file.  Default: session/reconstruction/mesh.ply.",
    )
    p.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default=None,
        help="Output PLY path.  Default: session/reconstruction/mesh_colored.ply.",
    )
    p.add_argument(
        "--cameras",
        metavar="NAME",
        nargs="+",
        default=None,
        help=(
            "Camera source names to use.  Defaults to all cameras found "
            "in session/images/.  Example: frontleft_fisheye_image back_fisheye_image"
        ),
    )
    p.add_argument(
        "--max-images",
        metavar="N",
        type=int,
        default=None,
        help="Maximum images to use per camera (for faster runs; default: all).",
    )
    p.add_argument(
        "--min-weight",
        metavar="W",
        type=float,
        default=0.05,
        help=(
            "Minimum accumulated view weight for a vertex to receive colour "
            "(default: 0.05).  Vertices below this threshold are grey."
        ),
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    session = args.session.resolve()
    if not session.exists():
        sys.exit(f"ERROR: session folder not found: {session}")

    try:
        out = colorize_mesh(
            session_path=session,
            mesh_path=args.mesh,
            output_path=args.out,
            cameras=args.cameras,
            max_images_per_camera=args.max_images,
            min_view_weight=args.min_weight,
        )
        print(f"\nColoured mesh written to:\n  {out}")
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")
    except Exception as e:
        logging.exception("Colorization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
