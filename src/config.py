"""Configuration resolver with CLI args > .env > interactive prompt precedence."""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for SPOT data capture and reconstruction."""

    # Robot connection
    robot_hostname: str
    robot_username: str
    robot_password: str

    # Output and processing
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "recordings")

    # Reconstruction parameters
    voxel_size: float = 0.05  # Meters, for downsampling
    loop_closure_threshold: float = 2.0  # Meters, spatial proximity for loop detection
    max_correspondence_distance: float = 0.1  # Meters, for registration
    icp_iterations: int = 50

    # Capture parameters
    lidar_sample_rate: float = 10.0  # Hz
    image_sample_rate: float = 5.0  # Hz (per camera)

    # AI Reconstruction parameters
    ai_model: str = "auto"             # "auto" | "mast3r" | "dust3r" | "vggt" | "geometric"
    ai_device: str = "auto"            # "auto" | "cuda" | "mps" | "cpu"
    ai_image_size: int = 512           # resize long edge to this (px)
    ai_max_images: int = 100           # keyframe cap
    ai_keyframe_interval: int = 5      # INTERVAL strategy: every Nth frame
    ai_voxel_size: float = 0.05        # post-processing downsample (m)
    ai_confidence_threshold: float = 1.5

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.log_file:
            self.log_file = Path(self.log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """
    Load configuration with precedence: CLI args > .env > interactive prompt.

    Args:
        args: Parsed CLI arguments (if None, parse from sys.argv)

    Returns:
        Config dataclass with resolved settings
    """
    if args is None:
        args = parse_args()

    # Step 1: Load .env file
    load_dotenv()

    # Step 2: Resolve hostname (CLI > .env > prompt)
    hostname = args.hostname or os.getenv("BOSDYN_HOSTNAME")
    if not hostname:
        hostname = input("Enter robot hostname [default: 192.168.10.3]: ").strip()
        if not hostname:
            hostname = "192.168.10.3"

    # Step 3: Resolve username (CLI > .env > prompt)
    username = args.username or os.getenv("BOSDYN_USERNAME")
    if not username:
        username = input("Enter robot username [default: student]: ").strip()
        if not username:
            username = "student"

    # Step 4: Resolve password (CLI > .env > prompt)
    password = args.password or os.getenv("BOSDYN_PASSWORD")
    if not password:
        import getpass

        password = getpass.getpass("Enter robot password: ")

    # Step 5: Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("recordings")

    # Step 6: Build config
    config = Config(
        robot_hostname=hostname,
        robot_username=username,
        robot_password=password,
        output_dir=output_dir,
        voxel_size=args.voxel_size,
        loop_closure_threshold=args.loop_closure_threshold,
        max_correspondence_distance=args.max_correspondence_distance,
        icp_iterations=args.icp_iterations,
        log_level=args.log_level,
    )

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SPOT LiDAR data capture and 3D reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Robot connection
    parser.add_argument(
        "-H",
        "--hostname",
        type=str,
        default=None,
        help="Robot hostname (overrides .env BOSDYN_HOSTNAME)",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        default=None,
        help="Robot username (overrides .env BOSDYN_USERNAME)",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        default=None,
        help="Robot password (overrides .env BOSDYN_PASSWORD)",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="recordings",
        help="Output directory for recorded sessions",
    )

    # Reconstruction parameters
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size (m) for downsampling point clouds",
    )
    parser.add_argument(
        "--loop-closure-threshold",
        type=float,
        default=2.0,
        help="Spatial proximity threshold (m) for loop closure detection",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=0.1,
        help="Max correspondence distance (m) for ICP registration",
    )
    parser.add_argument(
        "--icp-iterations",
        type=int,
        default=50,
        help="Number of ICP iterations",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()
