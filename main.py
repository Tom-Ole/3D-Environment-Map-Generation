"""Entry point for SPOT LiDAR capture and 3D reconstruction application."""
import sys

sys.dont_write_bytecode = True

from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gui.app import main

if __name__ == "__main__":
    main()
