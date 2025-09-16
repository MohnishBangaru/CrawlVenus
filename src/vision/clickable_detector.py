"""Clickable element detection for VisionEngine."""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from .models import BoundingBox, UIElement


class ClickableDetector:
    """Detect clickable/interactive UI elements from screenshots."""
    
    def __init__(self):
        """Initialize the clickable detector."""
        self.min_button_size = 30  # Minimum size for a button
        self.max_button_size = 300  # Maximum size for a button
        self.edge_threshold = 50  # Edge detection threshold
        
    def detect_clickable_elements(self, image_path: str) -> list[UIElement]:
        """Detect clickable elements from a screenshot.
        
        Parameters
        ----------
        image_path : str
            Path to the screenshot image
            
        Returns
        -------
        list[UIElement]
            List of detected clickable elements
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            clickable_elements = []
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 1. Detect button-like shapes (rectangles with borders)
            button_elements = self._detect_button_shapes(gray)
            clickable_elements.extend(button_elements)
            
            # 2. Detect colored interactive elements
            colored_elements = self._detect_colored_elements(hsv)
            clickable_elements.extend(colored_elements)
            
            # 3. Detect edge-bounded elements (likely clickable)
            edge_elements = self._detect_edge_bounded_elements(gray)
            clickable_elements.extend(edge_elements)
            
            # 4. Remove duplicates and overlapping elements
            clickable_elements = self._remove_duplicates(clickable_elements)
            
            logger.debug(f"ClickableDetector found {len(clickable_elements)} clickable elements")
            return clickable_elements
            
        except Exception as e:
            logger.warning(f"Clickable detection failed: {e}")
            return []
    
    def _detect_button_shapes(self, gray: np.ndarray) -> list[UIElement]:
        """Detect button-like rectangular shapes with borders."""
        elements = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if self.min_button_size <= w <= self.max_button_size and self.min_button_size <= h <= self.max_button_size:
                    # Check aspect ratio (not too long/short)
                    aspect_ratio = w / h
                    if 0.2 <= aspect_ratio <= 5.0:
                        bbox = BoundingBox(x, y, x + w, y + h)
                        elements.append(UIElement(
                            bbox=bbox,
                            text="button",
                            confidence=0.7,
                            element_type="button"
                        ))
        
        return elements
    
    def _detect_colored_elements(self, hsv: np.ndarray) -> list[UIElement]:
        """Detect elements with common interactive colors."""
        elements = []
        
        # Common interactive colors (buttons, links, etc.)
        color_ranges = [
            # Blue buttons/links
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # Red buttons
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            # Green buttons
            (np.array([40, 50, 50]), np.array([80, 255, 255])),
            # Orange buttons
            (np.array([10, 50, 50]), np.array([25, 255, 255])),
        ]
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours in the color mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if self.min_button_size <= w <= self.max_button_size and self.min_button_size <= h <= self.max_button_size:
                    bbox = BoundingBox(x, y, x + w, y + h)
                    elements.append(UIElement(
                        bbox=bbox,
                        text="colored_element",
                        confidence=0.6,
                        element_type="colored_button"
                    ))
        
        return elements
    
    def _detect_edge_bounded_elements(self, gray: np.ndarray) -> list[UIElement]:
        """Detect elements bounded by strong edges (likely interactive)."""
        elements = []
        
        # Edge detection
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        
        # Morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and edge density
            if self.min_button_size <= w <= self.max_button_size and self.min_button_size <= h <= self.max_button_size:
                # Calculate edge density in the region
                roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi > 0) / (w * h)
                
                if edge_density > 0.1:  # At least 10% of pixels are edges
                    bbox = BoundingBox(x, y, x + w, y + h)
                    elements.append(UIElement(
                        bbox=bbox,
                        text="edge_bounded",
                        confidence=0.5,
                        element_type="edge_bounded"
                    ))
        
        return elements
    
    def _remove_duplicates(self, elements: list[UIElement]) -> list[UIElement]:
        """Remove duplicate and overlapping elements."""
        if not elements:
            return []
        
        # Sort by confidence (highest first)
        elements = sorted(elements, key=lambda e: e.confidence, reverse=True)
        kept = []
        
        for element in elements:
            # Check if this element overlaps significantly with any kept element
            is_duplicate = False
            for kept_element in kept:
                iou = self._calculate_iou(element.bbox, kept_element.bbox)
                if iou > 0.3:  # More than 30% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(element)
        
        return kept
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        # Calculate intersection
        x_left = max(bbox1.left, bbox2.left)
        y_top = max(bbox1.top, bbox2.top)
        x_right = min(bbox1.right, bbox2.right)
        y_bottom = min(bbox1.bottom, bbox2.bottom)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = bbox1.width() * bbox1.height()
        bbox2_area = bbox2.width() * bbox2.height()
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


def detect_clickable_elements(image_path: str) -> list[UIElement]:
    """Convenience function to detect clickable elements."""
    detector = ClickableDetector()
    return detector.detect_clickable_elements(image_path) 