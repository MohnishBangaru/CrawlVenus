"""Vision debugging helpers: draw bounding boxes onto screenshots."""

from __future__ import annotations

import os

import cv2  # type: ignore

from ..core.config import config
from .models import UIElement


def save_debug_overlay(image_path: str, elements: list[UIElement]) -> None:
    """Draw bounding boxes on image and save under vision_debug directory."""
    if not config.save_vision_debug:
        return

    img = cv2.imread(image_path)
    if img is None:
        return

    for el in elements:
        color = (0, 0, 255)  # Red in BGR
        x1, y1, x2, y2 = el.bbox.as_tuple()

        # Draw thicker rectangle for better visibility
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)

        # Prepare text label
        label = f"{el.text}:{el.confidence:.2f}"
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)

        # Background rectangle (white) behind text for legibility
        text_bg_tl = (x1, max(0, y1 - text_h - 4))
        text_bg_br = (x1 + text_w + 4, max(0, y1))
        cv2.rectangle(img, text_bg_tl, text_bg_br, (255, 255, 255), thickness=cv2.FILLED)

        # Put red text on top
        text_org = (x1 + 2, max(10, y1 - 2))
        cv2.putText(
            img,
            label,
            text_org,
            font,
            font_scale,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    from pathlib import Path
    debug_dir = Path(image_path).resolve().parent.parent / config.ocr_images_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    base = Path(image_path).name.replace(".png", "_debug.png")
    cv2.imwrite(str(debug_dir / base), img) 