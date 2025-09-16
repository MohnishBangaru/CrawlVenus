"""Template matching utilities for VisionEngine."""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from functools import lru_cache

import cv2  # type: ignore
import numpy as np  # type: ignore
from loguru import logger

from ..core.config import config
from .models import BoundingBox, UIElement


@dataclass(slots=True)
class Template:
    """In-memory grayscale template with an associated human-readable name."""

    name: str
    image: np.ndarray  # grayscale template image


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


@lru_cache(maxsize=1)
def _load_templates() -> list[Template]:
    """Load all PNG/JPG templates from the templates directory (lazy)."""
    templates: list[Template] = []
    if not os.path.isdir(TEMPLATES_DIR):
        logger.debug("Template directory not found: {0}", TEMPLATES_DIR)
        return templates

    for fname in os.listdir(TEMPLATES_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(TEMPLATES_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Failed to load template {0}", path)
            continue
        name = os.path.splitext(fname)[0]
        templates.append(Template(name=name, image=img))
    logger.debug("Loaded {0} templates", len(templates))
    return templates


def match_templates(gray_screenshot: np.ndarray) -> list[UIElement]:
    """Return UIElement list for each matched template above threshold."""
    detected: list[UIElement] = []
    threshold = float(config.cv_template_matching_threshold)
    max_scales = int(config.cv_template_max_scales)
    scale_step = float(config.cv_template_scale_step)

    def _process_template(tpl: Template) -> list[UIElement]:
        local_detected: list[UIElement] = []
        try:
            scales = _generate_scales(max_scales, scale_step)
            matched_found = False
            for scale in scales:
                # resize
                if scale == 1.0:
                    tpl_resized = tpl.image
                else:
                    h0, w0 = tpl.image.shape[:2]
                    new_w, new_h = int(w0 * scale), int(h0 * scale)
                    if new_w < 8 or new_h < 8:
                        continue
                    tpl_resized = cv2.resize(
                        tpl.image,
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA,
                    )

                res = cv2.matchTemplate(gray_screenshot, tpl_resized, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                if loc[0].size == 0:
                    continue

                matched_found = True
                h, w = tpl_resized.shape[:2]
                for pt_y, pt_x in zip(*loc):
                    bbox = BoundingBox(pt_x, pt_y, pt_x + w, pt_y + h)
                    score = float(res[pt_y, pt_x])
                    local_detected.append(
                        UIElement(
                            bbox=bbox,
                            text=tpl.name,
                            confidence=score,
                            element_type="template",
                        )
                    )

                if matched_found:
                    break
        except Exception as exc:
            logger.warning("Template match failed for %s: %s", tpl.name, exc)
        return local_detected

    templates = _load_templates()
    if not templates:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.cv_template_threads) as executor:
        for res_list in executor.map(_process_template, templates):
            detected.extend(res_list)
    if detected:
        detected = _non_max_suppression(detected, iou_thresh=0.3)
        logger.debug("TemplateMatcher detected %d elements after NMS", len(detected))
    return detected


# ------------------------------------------------------------------
# Non-max suppression helpers
# ------------------------------------------------------------------


def _iou(box1: BoundingBox, box2: BoundingBox) -> float:  # Intersection-over-Union
    inter_left = max(box1.left, box2.left)
    inter_top = max(box1.top, box2.top)
    inter_right = min(box1.right, box2.right)
    inter_bottom = min(box1.bottom, box2.bottom)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    box1_area = box1.width() * box1.height()
    box2_area = box2.width() * box2.height()
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0.0


def _non_max_suppression(elements: list[UIElement], *, iou_thresh: float) -> list[UIElement]:
    """Suppress overlapping detections keeping highest-confidence element."""
    if not elements:
        return []

    # Sort by confidence descending
    elems = sorted(elements, key=lambda e: e.confidence, reverse=True)
    kept: list[UIElement] = []

    while elems:
        current = elems.pop(0)
        kept.append(current)
        elems = [e for e in elems if _iou(current.bbox, e.bbox) < iou_thresh]

    return kept


# ------------------------------------------------------------------
# Scale helpers
# ------------------------------------------------------------------


def _generate_scales(max_scales: int, step: float) -> list[float]:
    """Return list of scale factors centred at 1.0 (e.g., [1.0, 0.9,1.1,0.81,...])."""
    scales: list[float] = [1.0]
    down = 1.0
    up = 1.0
    for _ in range(max_scales):
        down *= step
        up /= step
        scales.extend([down, up])
    return scales


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------

def reload_templates() -> None:
    """Clear template cache so new images are picked up without restart."""
    _load_templates.cache_clear() 