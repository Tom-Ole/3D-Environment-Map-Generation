"""Shared dataclasses and enumerations for the AI reconstruction pipeline."""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np


class ModelType(str, Enum):
    AUTO = "auto"
    MAST3R = "mast3r"
    DUST3R = "dust3r"
    VGGT = "vggt"
    GEOMETRIC = "geometric"


class DeviceType(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class KeyframeStrategy(str, Enum):
    INTERVAL = "interval"
    MOTION = "motion"


@dataclass
class AIReconstructionConfig:
    """Full configuration for the AI reconstruction pipeline."""

    # Model selection
    model_type: ModelType = ModelType.AUTO
    device: DeviceType = DeviceType.AUTO

    # Image handling
    image_size: int = 512               # resize long edge to this (px)
    camera_sources: List[str] = field(default_factory=lambda: [
        "frontleft_fisheye_image",
        "frontright_fisheye_image",
    ])

    # Keyframe selection
    keyframe_strategy: KeyframeStrategy = KeyframeStrategy.INTERVAL
    keyframe_interval: int = 5          # every Nth image (INTERVAL mode)
    keyframe_min_translation: float = 0.3    # metres (MOTION mode)
    keyframe_min_rotation_deg: float = 10.0  # degrees (MOTION mode)
    max_images: int = 100               # hard cap after keyframe selection

    # Model-specific tuning
    confidence_threshold: float = 1.5   # DUSt3R/MASt3R confidence mask
    global_alignment_iter: int = 300    # DUSt3R global aligner iterations

    # Post-processing
    voxel_size: float = 0.05           # downsampling voxel (m); 0 = skip
    min_depth: float = 0.1             # z-filter lower bound (m)
    max_depth: float = 50.0            # z-filter upper bound (m)

    # Meshing (Poisson surface reconstruction)
    mesh_enabled: bool = True          # build a surface mesh from the cloud
    poisson_depth: int = 10            # octree depth; higher = finer + slower
    mesh_density_quantile: float = 0.02  # trim this fraction of lowest-density verts
    normal_radius: float = 0.1         # normal-estimation radius (m, metric scale)


@dataclass
class ImageRecord:
    """Metadata for a single captured camera image."""

    path: Path
    source_name: str       # e.g. "frontleft_fisheye_image"
    frame_id: int
    inferred_timestamp: float = 0.0
    camera_idx: int = 0    # per-source sequential index


@dataclass
class AIReconstructionProgress:
    """Progress report emitted by the pipeline after each stage step."""

    stage: str
    stage_index: int
    total_stages: int
    stage_pct: float    # 0–100 within current stage
    overall_pct: float  # 0–100 across all stages
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AIPointCloudResult:
    """
    Raw output from a reconstruction model.

    Coordinates are in the model's internal scene frame.  For models that
    produce metric-scale output (MASt3R, Geometric) the values are in metres.
    Non-metric models (DUSt3R, VGGT) produce an arbitrary scale.
    """

    points: np.ndarray                           # Nx3 float32, world-frame 3-D points
    colors: Optional[np.ndarray] = None          # Nx3 uint8 RGB
    confidence: Optional[np.ndarray] = None      # N float32, higher = better
    camera_poses: Optional[np.ndarray] = None    # Mx4x4 float64 camera-to-world
    depth_maps: Optional[List[np.ndarray]] = None
    image_paths: Optional[List[Path]] = None
    model_name: str = ""
    metric_scale: bool = False


@dataclass
class AIReconstructionResult:
    """Final output returned by AIReconstructionPipeline.run()."""

    point_cloud_path: Optional[Path] = None
    mesh_path: Optional[Path] = None
    camera_poses_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    point_count: int = 0
    keyframe_count: int = 0
    image_count: int = 0
    model_used: str = ""
    device_used: str = ""
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str = ""
