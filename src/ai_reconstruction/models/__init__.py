"""Model registry: discover, instantiate, and rank reconstruction models."""

import logging
from typing import List, Optional, Tuple

from ai_reconstruction.models.base import ReconstructionModel

logger = logging.getLogger(__name__)

# Preference order for "auto" selection: best quality first, fallback last.
_AUTO_PRIORITY = ["mast3r", "dust3r", "vggt", "geometric"]


def get_available_models() -> List[Tuple[str, bool, str]]:
    """
    Return availability status for all registered models.

    Returns:
        List of (name, is_available, description)
    """
    from ai_reconstruction.models.mast3r_model import MASt3RModel
    from ai_reconstruction.models.dust3r_model import DUSt3RModel
    from ai_reconstruction.models.vggt_model import VGGTModel
    from ai_reconstruction.models.geometric_model import GeometricModel

    return [
        ("mast3r", MASt3RModel.is_available(), MASt3RModel.description),
        ("dust3r", DUSt3RModel.is_available(), DUSt3RModel.description),
        ("vggt", VGGTModel.is_available(), VGGTModel.description),
        ("geometric", GeometricModel.is_available(), GeometricModel.description),
    ]


def get_model(
    model_type: str = "auto",
    device: str = "auto",
    **kwargs,
) -> Optional[ReconstructionModel]:
    """
    Instantiate a reconstruction model by name.

    Args:
        model_type: "auto" | "mast3r" | "dust3r" | "vggt" | "geometric"
        device:     "auto" | "cuda" | "mps" | "cpu"
        **kwargs:   Model-specific constructor parameters (e.g. spot_poses,
                    intrinsics, image_size for GeometricModel)

    Returns:
        Instantiated ReconstructionModel, or None if unavailable.
    """
    from ai_reconstruction.models.mast3r_model import MASt3RModel
    from ai_reconstruction.models.dust3r_model import DUSt3RModel
    from ai_reconstruction.models.vggt_model import VGGTModel
    from ai_reconstruction.models.geometric_model import GeometricModel

    registry = {
        "mast3r": MASt3RModel,
        "dust3r": DUSt3RModel,
        "vggt": VGGTModel,
        "geometric": GeometricModel,
    }

    if model_type == "auto":
        for name in _AUTO_PRIORITY:
            cls = registry[name]
            if cls.is_available():
                logger.info(f"Auto-selected model: {name}")
                # GeometricModel accepts extra kwargs; others ignore them safely
                return _construct(cls, name, device, kwargs)
        logger.error(
            "No reconstruction model available. "
            "Install mast3r, dust3r, vggt, or opencv-python."
        )
        return None

    cls = registry.get(model_type)
    if cls is None:
        logger.error(f"Unknown model type '{model_type}'. "
                     f"Valid: {list(registry)}")
        return None

    if not cls.is_available():
        logger.error(
            f"Model '{model_type}' is not available (missing dependencies). "
            f"See model docstring for install instructions."
        )
        return None

    return _construct(cls, model_type, device, kwargs)


def _construct(cls, name: str, device: str, kwargs: dict) -> ReconstructionModel:
    """Construct a model, passing only the kwargs its __init__ accepts."""
    import inspect

    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}

    safe_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    skipped = set(kwargs) - set(safe_kwargs)
    if skipped:
        logger.debug(f"[{name}] Ignoring unsupported kwargs: {skipped}")

    return cls(device=device, **safe_kwargs)
