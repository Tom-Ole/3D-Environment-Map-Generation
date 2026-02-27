from dataclasses import dataclass
from typing import Optional
import argparse
from pathlib import Path


@dataclass
class ParsedArgs:
    """
    Container for all command-line arguments.
    """
    dataset_path: str
    sequence_name: str
    voxel_size: float
    visualize: bool
    deep_feature_mapping: bool
    num_frames: Optional[int]
    output_mesh_path: str


def parse_path(
    path_str: str,
    *,
    must_exist: bool = False,
    create_parent: bool = False,
) -> str:

    try:
        path = Path(path_str).expanduser().resolve(strict=False)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid path: {path_str}") from e

    if must_exist and not path.exists():
        raise argparse.ArgumentTypeError(
            f"Path does not exist: {path}"
        )

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)

def parse_args() -> ParsedArgs:

    parser = argparse.ArgumentParser(
        description="3D Reconstruction and Environment Map Generation Pipeline"
    )

    parser.add_argument(
        "-d",
        "--dataset-path",
        required=True,
        type=str,
        help="Path to the dataset directory (e.g. ./dataset/sun3d).",
    )

    parser.add_argument(
        "-n",
        "--sequence-name",
        required=True,
        type=str,
        help="Name of the sequence to process (e.g. seq-01).",
    )

    parser.add_argument(
        "-s",
        "--voxel-size",
        default=0.05,
        type=float,
        help="Voxel size in meters for TSDF integration (default: 0.05).",
    )

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Enable real-time visualization. (default: True)",
    )

    parser.add_argument(
        "--deep-feature-mapping",
        action="store_true",
        help="Enable deep feature-based mapping. (default: false)",
    )

    parser.add_argument(
        "-f",
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to process. If omitted, all frames are processed. (default: None)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="./output/mesh.obj",
        type=str,
        help="Output path for the generated mesh (default: ./output/mesh.obj).",
    )

    args = parser.parse_args()
    
    input_path = parse_path(args.dataset_path, must_exist=True)
    output_path = parse_path(args.output, create_parent=True)

    return ParsedArgs(
        dataset_path=input_path,
        sequence_name=args.sequence_name,
        voxel_size=args.voxel_size,
        visualize=args.visualize,
        deep_feature_mapping=args.deep_feature_mapping,
        num_frames=args.num_frames,
        output_mesh_path=output_path,
    )