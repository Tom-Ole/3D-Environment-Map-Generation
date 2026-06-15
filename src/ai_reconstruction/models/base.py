"""Abstract base class for all AI and geometric reconstruction models."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

from ai_reconstruction.types import AIPointCloudResult

logger = logging.getLogger(__name__)


class ReconstructionModel(ABC):
    """
    Common interface for every reconstruction model.

    Concrete models implement is_available(), load(), and reconstruct().
    The pipeline calls them in that order; load() may download weights.
    """

    name: str = "base"
    description: str = ""

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._loaded = False

    # ── Abstract methods ──────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when all required packages are importable."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load or download model weights.  Called once before reconstruct()."""
        ...

    @abstractmethod
    def reconstruct(
        self,
        image_paths: List[Path],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> AIPointCloudResult:
        """
        Reconstruct a 3-D scene from the given images.

        Args:
            image_paths: Ordered list of input image paths (keyframes)
            progress_cb: Optional (pct 0-100, message) callback for UI updates

        Returns:
            AIPointCloudResult with at minimum a points array.
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve "auto" to the best available PyTorch device."""
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _report(
        self,
        cb: Optional[Callable[[float, str], None]],
        pct: float,
        msg: str,
    ) -> None:
        """Emit a progress callback and log at INFO level."""
        logger.info(f"[{self.name}] {pct:.0f}%  {msg}")
        if cb:
            try:
                cb(pct, msg)
            except Exception:
                pass
