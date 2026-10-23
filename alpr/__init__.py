"""ALPR package: plate detection + OCR + tracking + watchlist matching."""

from .pipeline import PlatePipeline, PlateReading
from .tracker import PlateTracker, Track
from .matcher import PlateMatcher

__all__ = [
    "PlatePipeline",
    "PlateReading",
    "PlateTracker",
    "Track",
    "PlateMatcher",
]
