
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


OUTPUT_DIR = Path("recordings")


def find_latest_session(output_dir: Path) -> Path:
    sessions = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not sessions:
        print(f"No sessions found in {output_dir}")
        sys.exit(1)
    return sessions[-1]


def colorize_by_height(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    z = pts[:, 2]
    z_norm = (z - z.min()) / (z.ptp() + 1e-6)
    # Blue (low) → green → red (high)
    colors = np.zeros((len(pts), 3))
    colors[:, 0] = z_norm          # R increases with height
    colors[:, 1] = 1.0 - z_norm   # G decreases with height
    colors[:, 2] = 0.5 * (1.0 - z_norm)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def load_frames(lidar_dir: Path, frame_idx: int | None) -> list[o3d.geometry.PointCloud]:
    ply_files = sorted(lidar_dir.glob("*.ply"))
    if not ply_files:
        print(f"No .ply files found in {lidar_dir}")
        sys.exit(1)

    if frame_idx is not None:
        if frame_idx >= len(ply_files):
            print(f"Frame {frame_idx} out of range (0–{len(ply_files)-1})")
            sys.exit(1)
        ply_files = [ply_files[frame_idx]]

    clouds = []
    for f in ply_files:
        pcd = o3d.io.read_point_cloud(str(f))
        if len(pcd.points) > 0:
            clouds.append(pcd)

    print(f"Loaded {len(clouds)} frame(s), "
          f"{sum(len(c.points) for c in clouds):,} total points")
    return clouds


def make_pose_lineset(poses_path: Path) -> o3d.geometry.LineSet | None:
    if not poses_path.exists():
        return None
    poses = np.load(poses_path)  # (N, 8): timestamp x y z qx qy qz qw
    if len(poses) < 2:
        return None
    xyz = poses[:, 1:4].astype(np.float64)
    lines = [[i, i + 1] for i in range(len(xyz) - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(xyz),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color([1.0, 0.8, 0.0])  # yellow path
    return ls


def main():
    parser = argparse.ArgumentParser(description="Visualize LiDAR session")
    parser.add_argument("session", nargs="?", help="Path to session folder")
    parser.add_argument("--frame", type=int, default=None, help="Show a single frame index")
    parser.add_argument("--poses", action="store_true", help="Overlay robot path")
    args = parser.parse_args()

    session_dir = Path(args.session) if args.session else find_latest_session(OUTPUT_DIR)
    lidar_dir = session_dir / "lidar"

    if not lidar_dir.exists():
        print(f"No lidar/ folder in {session_dir}")
        sys.exit(1)

    print(f"Session: {session_dir.name}")

    clouds = load_frames(lidar_dir, args.frame)

    # Merge all frames into one cloud
    merged = o3d.geometry.PointCloud()
    for c in clouds:
        merged += c

    colorize_by_height(merged)

    geometries = [merged]

    if args.poses:
        ls = make_pose_lineset(session_dir / "poses.npy")
        if ls:
            geometries.append(ls)
            print("Robot path overlaid in yellow")
        else:
            print("No poses.npy found, skipping path")

    # Add a small coordinate frame at the origin
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    print("Controls: left-drag=rotate  scroll=zoom  shift+drag=pan  Q=quit")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"LiDAR — {session_dir.name}",
        width=1280,
        height=720,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
