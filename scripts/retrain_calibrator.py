"""
Run from btip-gridlock2/ root:
    python scripts/retrain_calibrator.py
"""
import sys
import os
import logging

# Add repo root to path so 'backend' package is findable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

from backend.models.risk.calibration import train_calibrator
train_calibrator()
print("Calibrator saved successfully with correct module path.")