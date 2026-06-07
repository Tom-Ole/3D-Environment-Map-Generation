# SPOT LiDAR 3D Capture & Reconstruction

A desktop application for capturing multi-modal sensor data from a Boston Dynamics SPOT robot (LiDAR + body cameras + IMU) during manual teleoperation, then performing offline 3D reconstruction to produce globally-consistent, metric, colored point clouds and textured meshes.

## Features

- **Live Data Capture**: Stream LiDAR scans, camera images, and robot poses to disk in real-time
- **Read-Only Operation**: No motion lease required; works during manual teleoperation
- **Offline Reconstruction Pipeline**:
  - KISS-ICP odometry estimation
  - SPOT vision-frame pose seeding
  - Loop closure detection and registration
  - Global pose graph optimization
  - Point cloud fusion and voxel downsampling
  - Poisson mesh generation
  - Height-based colorization (expandable to camera projection)
- **Desktop GUI**: PySide6-based interface with status monitoring and progress visualization

## System Requirements

### Python Version

**Target: Python 3.12** (conservative, all dependencies stable)

**Rationale**: As of February 2025, the key dependencies (bosdyn-api, open3d, kiss-icp) have stable wheels for Python 3.12. Python 3.14 wheels are not yet published for these libraries; upgrading will be possible once upstream maintainers release 3.14 wheels, likely in 2025-2026.

### OS & Hardware

- Linux (Ubuntu 20.04 LTS or later recommended)
- Dual-core CPU minimum; quad-core recommended for reconstruction
- 8 GB RAM minimum; 16 GB recommended
- Solid-state disk with ~50 GB free space per session

### Network

- Network connectivity to SPOT robot (direct or via gateway)
- Standard Python package index (PyPI) access for initial setup

## Installation

### 1. Create a Virtual Environment

```bash
cd /path/to/3D-Environment-Map-Generation
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- bosdyn SDK (for robot communication)
- open3d (for 3D geometry)
- kiss-icp (for odometry)
- PySide6 (for GUI)
- numpy, scipy, opencv-python (supporting libraries)
- python-dotenv (for configuration)

### 3. Configure Credentials

Copy the example environment file and fill in your robot's details:

```bash
cp .env.example .env
# Edit .env with your robot hostname, username, and password
```

**File: `.env`**
```
BOSDYN_HOSTNAME=192.168.10.3
BOSDYN_USERNAME=student
BOSDYN_PASSWORD=your_robot_password
```

Alternatively, pass credentials via CLI arguments (see Usage below).

## Usage

### Launch the GUI Application

```bash
python main.py
```

Or with command-line overrides:

```bash
python main.py \
  --hostname 192.168.10.3 \
  --username student \
  --password your_password \
  --output-dir ./recordings
```

### Command-Line Arguments

```
-H, --hostname          Robot hostname [default: 192.168.10.3]
-u, --username          Robot username [default: from .env]
-p, --password          Robot password [default: from .env]
-o, --output-dir        Output directory for sessions [default: recordings/]
--voxel-size            Voxel size for downsampling (meters) [default: 0.05]
--loop-closure-threshold Spatial threshold for loop detection (m) [default: 2.0]
--max-correspondence-distance ICP correspondence distance (m) [default: 0.1]
--icp-iterations        ICP iterations [default: 50]
--log-level             DEBUG|INFO|WARNING|ERROR [default: INFO]
```

### GUI Workflow

#### 1. Capture Tab

1. Enter robot hostname (or use `.env` default)
2. Click **"Connect"** to authenticate and test connection
3. Click **"Start Recording"** to begin capturing LiDAR + images + poses
4. Click **"Stop Recording"** when done
5. Data is saved to `recordings/<YYYYMMDD_HHMMSS>/`

#### 2. Reconstruct Tab

1. Click **"Browse..."** to select a session folder
2. Adjust reconstruction parameters if needed:
   - **Voxel Size**: Larger = faster but coarser (0.05m typical)
   - **Loop Threshold**: Max distance for loop closure candidates (2.0m typical)
3. Click **"Run Reconstruction"** to start the pipeline
4. Monitor progress in the log window
5. Results are saved to `session/reconstruction/`:
   - `cloud_optimized.ply` — Colored point cloud
   - `mesh.ply` — Mesh (PLY format)
   - `mesh.obj` — Mesh (OBJ format, for external viewers)

## Data Layout

### Recording Session Folder Structure

```
recordings/20260607_140000/                 # Session ID = timestamp
├── metadata.json                          # Session metadata
├── poses.npy                              # Nx7 array [t, x, y, z, qx, qy, qz, qw]
├── intrinsics.json                        # Camera intrinsics (per source)
├── lidar/
│   ├── 00000.ply                          # Colored point cloud (intensity as color)
│   ├── 00000_raw.npy                      # Raw Nx5 [x, y, z, intensity, ...]
│   ├── 00001.ply
│   └── ...
├── images/
│   ├── 00000_back_fisheye_image.png
│   ├── 00000_left_fisheye_image.png
│   └── ...
└── reconstruction/                        # Output of offline pipeline
    ├── cloud_optimized.ply                # Final colored point cloud
    ├── mesh.ply
    ├── mesh.obj
    └── log.txt                            # Pipeline execution log
```

### File Formats

- **Poses**: NumPy `.npy` file (Nx7 float32 array)
  - Columns: timestamp, x, y, z, qx, qy, qz, qw (quaternion scalar-last)
  - Frame: SPOT vision frame (typically)
  
- **LiDAR**: PLY format (text or binary, readable by CloudCompare, Meshlab)
  
- **Images**: PNG (lossless) with metadata in JSON

- **Intrinsics**: JSON with per-camera fx, fy, cx, cy, distortion coefficients

## Architecture

### Modules

- **config.py** — Configuration resolver (CLI → .env → interactive prompt)
- **capture/** — SPOT SDK wrappers (LiDAR, images, poses, IMU)
- **recording/** — Session management and thread-safe disk I/O
- **reconstruction/** — Offline pipeline (odometry, loop closure, optimization, meshing)
- **gui/** — PySide6 application (tabs, workers, dialogs)
- **utils/** — Transform utilities, timestamp synchronization, logging

### Reconstruction Pipeline

1. **Load Session** → Read scans, poses, intrinsics
2. **Odometry** → KISS-ICP with SPOT pose seeding
3. **Loop Closure** → Spatial proximity detection + ICP registration
4. **Global Optimization** → Open3D pose graph (Levenberg-Marquardt)
5. **Fusion** → Re-fuse all scans at optimized poses, voxel downsample
6. **Meshing** → Poisson mesh generation from fused cloud
7. **Export** → Save colored cloud + mesh (PLY + OBJ)

## Known Limitations & TODOs

- **KISS-ICP Integration**: Placeholder wrapper; full SDK integration pending verification on robot
- **Camera Colorization**: Currently height-based; full camera projection not yet implemented
- **Intrinsics Retrieval**: Assumes intrinsics come from ImageClient; **verify frame names and format on robot**
- **Loop Closure**: Uses small_gicp placeholder; optimize registration parameters for LiDAR
- **GUI Polish**: Basic functionality; no real-time 3D preview or mesh viewer yet
- **Performance**: Reconstruction pipeline is single-threaded; can be parallelized for submaps

## Troubleshooting

### Connection Fails

```
Error: Failed to connect to robot
```

- Verify robot hostname/IP is reachable: `ping 192.168.10.3`
- Check credentials in `.env` or CLI args
- Ensure network allows gRPC (port 50051 by default)

### No LiDAR Scans Found

```
Error: No LiDAR scans found in session
```

- **TODO**: Verify `service_name="velodyne"` in `capture/lidar_client.py` matches your robot
- Check robot has EAP payload with Velodyne-16 enabled
- Review robot logs for PointCloudService errors

### Reconstruction Takes Long Time

- Reduce `--voxel-size` (coarser = faster)
- Increase `--icp-iterations` if registration quality is poor
- Run on a machine with more CPU cores; consider GPU support (future)

## Development Notes

### Adding Custom Reconstruction Steps

Each step in `reconstruction/` is modular:
- Edit `pipeline.py` to add new steps
- Follow the pattern: load data → process → emit progress → save output
- Register callbacks for GUI integration

### Extending Sensor Support

New sensor types can be added under `capture/`:
- Create a new `*_client.py` wrapper
- Follow the type definitions in `capture/types.py`
- Update `recording/writer.py` to handle new data

### GUI Customization

- Tabs are in `gui/tabs/`
- Workers in `gui/workers/`
- Widgets in `gui/widgets/`
- Use PySide6 signals/slots for thread-safe communication

## References

- [Boston Dynamics SDK Docs](https://dev.bostondynamics.com/)
- [Open3D Geometry Processing](http://www.open3d.org/)
- [KISS-ICP (LiDAR Odometry)](https://github.com/IbisRobotics/KISS-ICP)
- [PySide6 Documentation](https://doc.qt.io/qtforpython-6/)

## License

See LICENSE file (if applicable).

## Contact

For questions or issues, please refer to the project repository.
