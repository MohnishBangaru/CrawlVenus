"""Computer vision utilities for DroidBot-GPT.

This sub-package provides screenshot analysis, UI element detection, and OCR
capabilities used to turn raw screenshots into structured device state.
"""

from .engine import VisionEngine
from .models import BoundingBox, UIElement

__all__ = [
    "BoundingBox",
    "UIElement",
    "VisionEngine",
] 