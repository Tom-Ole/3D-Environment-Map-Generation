"""Entry point for SPOT LiDAR capture and 3D reconstruction application."""
import sys

#sys.dont_write_bytecode = True

from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Add locally cloned MASt3R + DUSt3R submodule to sys.path so that
# `import mast3r` resolves to mast3r/mast3r/ (the real Python package)
# rather than the repo root being picked up as a namespace package.
_mast3r_repo = Path(__file__).parent / "mast3r"
if _mast3r_repo.is_dir() and (_mast3r_repo / "mast3r" / "__init__.py").is_file():
    sys.path.insert(0, str(_mast3r_repo))
    _dust3r_repo = _mast3r_repo / "dust3r"
    if _dust3r_repo.is_dir() and (_dust3r_repo / "dust3r" / "__init__.py").is_file():
        sys.path.insert(0, str(_dust3r_repo))

from gui.app import main

if __name__ == "__main__":
    main()
