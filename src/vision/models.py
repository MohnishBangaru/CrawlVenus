"""Data models for computer vision subsystem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned rectangle (left, top, right, bottom) in pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    def width(self) -> int:
        """Width in pixels."""
        return self.right - self.left

    def height(self) -> int:
        """Height in pixels."""
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return bounding box as ``(left, top, right, bottom)`` tuple."""
        return self.left, self.top, self.right, self.bottom


@dataclass(slots=True)
class UIElement:
    """Representation of a detected UI element on screen."""

    bbox: BoundingBox
    text: str
    confidence: float
    element_type: str = "text"  # could be button, label, etc. 