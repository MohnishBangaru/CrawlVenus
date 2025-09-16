"""Vision engine for screenshot analysis and UI element detection."""

from __future__ import annotations

import concurrent.futures
import os

import cv2  # type: ignore
import pytesseract  # type: ignore
from loguru import logger

from ..core.config import config
# UI-Venus grounding engine
from .venus_ground import get_ui_venus_grounder
from .models import BoundingBox, UIElement


class VisionEngine:
    """Analyze screenshots and return detected UI elements using OCR."""

    def __init__(self, tesseract_lang: str = "eng") -> None:
        """Initialize VisionEngine.

        Parameters
        ----------
        tesseract_lang : str
            Tesseract language code (default is ``"eng"``).

        """
        # If the user provided a custom tesseract cmd path, set it.
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = tesseract_lang
        # Use a thread pool for OCR to avoid blocking event loop.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Check if OCR should be enabled and Tesseract is present
        self._ocr_available: bool = False
        if not config.use_ocr:
            logger.warning("VisionEngine: OCR disabled via configuration (USE_OCR=false)")
        else:
            try:
                # Try to find Tesseract in common locations
                tesseract_paths = [
                    "/opt/homebrew/bin/tesseract",  # macOS Homebrew
                    "/usr/local/bin/tesseract",     # macOS/Linux
                    "/usr/bin/tesseract",           # Linux
                    "/usr/local/lib/python3.11/dist-packages/tesseract",  # Python packages
                    "/usr/local/lib/python3.11/site-packages/tesseract",  # Python site-packages
                    "/usr/lib/python3.11/dist-packages/tesseract",        # System Python packages
                    "/usr/lib/python3.11/site-packages/tesseract",        # System Python site-packages
                    "/opt/conda/bin/tesseract",     # Conda installation
                    "/opt/conda/envs/*/bin/tesseract",  # Conda environments
                    "tesseract"                     # PATH
                ]
                
                tesseract_found = False
                
                # Add dynamic Python package directory search
                import sys
                for site_packages in sys.path:
                    if 'site-packages' in site_packages or 'dist-packages' in site_packages:
                        tesseract_paths.append(f"{site_packages}/tesseract")
                        tesseract_paths.append(f"{site_packages}/tesseract_ocr")
                
                # Add conda environment paths
                conda_prefix = os.getenv('CONDA_PREFIX')
                if conda_prefix:
                    tesseract_paths.append(f"{conda_prefix}/bin/tesseract")
                
                for path in tesseract_paths:
                    try:
                        # Handle wildcard paths for conda environments
                        if '*' in path:
                            import glob
                            expanded_paths = glob.glob(path)
                            for expanded_path in expanded_paths:
                                try:
                                    pytesseract.pytesseract.tesseract_cmd = expanded_path
                                    pytesseract.get_tesseract_version()
                                    logger.info(f"VisionEngine: Tesseract found at {expanded_path}")
                                    tesseract_found = True
                                    break
                                except Exception:
                                    continue
                            if tesseract_found:
                                break
                        else:
                            pytesseract.pytesseract.tesseract_cmd = path
                            pytesseract.get_tesseract_version()
                            logger.info(f"VisionEngine: Tesseract found at {path}")
                            tesseract_found = True
                            break
                    except Exception:
                        continue
                
                if tesseract_found:
                    self._ocr_available = True
                else:
                    # Try to use Python tesseract package as fallback
                    try:
                        import tesseract
                        logger.info("VisionEngine: Using Python tesseract package as fallback")
                        self._ocr_available = True
                        # Set a dummy path for pytesseract
                        pytesseract.pytesseract.tesseract_cmd = "python_tesseract_fallback"
                    except (ImportError, SyntaxError):
                        try:
                            import tesseract_ocr
                            logger.info("VisionEngine: Using Python tesseract_ocr package as fallback")
                            self._ocr_available = True
                            # Set a dummy path for pytesseract
                            pytesseract.pytesseract.tesseract_cmd = "python_tesseract_ocr_fallback"
                        except (ImportError, SyntaxError):
                            logger.warning(
                                "VisionEngine: Tesseract not found in common locations. OCR will be skipped."
                            )
                    
            except Exception as exc:
                logger.warning(
                    f"VisionEngine: Unexpected error checking Tesseract ({exc}). OCR disabled."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image_path: str) -> list[UIElement]:
        """Return list of detected `UIElement` objects from the screenshot."""
        # Short-circuit: use UI-Venus grounding if enabled
        if config.use_ui_venus:
            return get_ui_venus_grounder().detect_ui_elements(image_path)

        if not self._ocr_available:
            return []  # OCR disabled and no UI-Venus, return empty list

        if not os.path.exists(image_path):
            logger.error(f"Screenshot not found: {image_path}")
            return []

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []

        # Pre-process: convert to grayscale, optional resize for speed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple threshold to improve OCR contrast
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Run OCR asynchronously
        try:
            future = self._executor.submit(
                pytesseract.image_to_data,
                thresh,
                lang=self.lang,
                output_type=pytesseract.Output.DICT,
            )
            ocr_data = future.result()
        except Exception as e:
            # Fallback to Python tesseract package if pytesseract fails
            logger.warning(f"pytesseract failed: {e}, trying Python tesseract fallback")
            try:
                import tesseract
                # Use Python tesseract package directly
                ocr_data = self._run_python_tesseract_fallback(thresh)
            except (ImportError, SyntaxError):
                try:
                    import tesseract_ocr
                    ocr_data = self._run_python_tesseract_ocr_fallback(thresh)
                except (ImportError, SyntaxError):
                    logger.error("All OCR methods failed")
                    return []

        elements: list[UIElement] = []
        n_boxes = len(ocr_data["level"])
        for i in range(n_boxes):
            text = ocr_data["text"][i].strip()
            conf = float(ocr_data["conf"][i])
            if not text or conf < 0:
                continue  # skip empty / low confidence entries
            x, y, w, h = (
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["width"][i],
                ocr_data["height"][i],
            )
            bbox = BoundingBox(x, y, x + w, y + h)
            elements.append(UIElement(bbox=bbox, text=text, confidence=conf / 100.0))

        logger.debug(f"VisionEngine detected {len(elements)} text elements")

        # Template matching (non-text elements)
        try:
            from .template_matcher import match_templates
            tmpl_elements = match_templates(gray)
            elements.extend(tmpl_elements)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Template matching failed: {exc}")

        # Clickable element detection
        try:
            from .clickable_detector import detect_clickable_elements
            clickable_elements = detect_clickable_elements(image_path)
            elements.extend(clickable_elements)
            logger.debug(f"VisionEngine detected {len(clickable_elements)} clickable elements")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Clickable detection failed: {exc}")

        # Filter for interactive elements only
        interactive_elements = self._filter_interactive_elements(elements)
        
        # Save debug overlay if enabled
        try:
            from .debug import save_debug_overlay
            save_debug_overlay(image_path, interactive_elements)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to save debug overlay: {exc}")

        logger.debug(f"VisionEngine total interactive elements: {len(interactive_elements)}")
        return interactive_elements
    
    def _filter_interactive_elements(self, elements: list[UIElement]) -> list[UIElement]:
        """Filter elements to only include interactive/clickable ones.
        
        Parameters
        ----------
        elements : list[UIElement]
            List of all detected elements
            
        Returns
        -------
        list[UIElement]
            List of interactive elements only
        """
        interactive_elements = []
        
        for element in elements:
            # Include elements that are likely interactive
            if element.element_type in ["button", "colored_button", "edge_bounded", "template"]:
                # Only include if they meet size criteria
                bbox = element.bbox
                if 30 <= bbox.width() <= 300 and 30 <= bbox.height() <= 200:
                    interactive_elements.append(element)
            elif element.element_type == "text":
                # For text elements, check if they might be clickable
                # Look for common interactive text patterns
                text_lower = element.text.lower()
                interactive_keywords = [
                    "button", "click", "tap", "press", "select", "choose",
                    "ok", "cancel", "yes", "no", "save", "delete", "edit",
                    "add", "remove", "next", "back", "continue", "skip",
                    "login", "sign", "register", "submit", "send", "search",
                    "menu", "settings", "profile", "help", "close", "exit"
                ]
                
                # Check if text contains interactive keywords
                if any(keyword in text_lower for keyword in interactive_keywords):
                    interactive_elements.append(element)
                # Check if text is short and in button-like size (likely a button label)
                elif len(element.text) <= 15 and element.confidence > 0.8:
                    # Check if it's in a reasonable size range for a button
                    bbox = element.bbox
                    if 30 <= bbox.width() <= 200 and 20 <= bbox.height() <= 80:
                        interactive_elements.append(element)
        
        # Remove duplicates and overlapping elements
        interactive_elements = self._remove_overlapping_elements(interactive_elements)
        
        logger.debug(f"Filtered {len(elements)} total elements to {len(interactive_elements)} interactive elements")
        return interactive_elements
    
    def _remove_overlapping_elements(self, elements: list[UIElement]) -> list[UIElement]:
        """Remove overlapping elements, keeping the most likely interactive ones."""
        if not elements:
            return []
        
        # Sort by confidence and element type priority
        def element_priority(element: UIElement) -> int:
            priority_map = {
                "button": 1,
                "colored_button": 2,
                "text": 3,
                "edge_bounded": 4,
                "template": 5
            }
            return priority_map.get(element.element_type, 6)
        
        elements = sorted(elements, key=lambda e: (element_priority(e), e.confidence), reverse=True)
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
    
    def _run_python_tesseract_fallback(self, image) -> dict:
        """Fallback method using Python tesseract package."""
        try:
            import tesseract
            # Convert image to format expected by Python tesseract
            import numpy as np
            from PIL import Image
            
            # Convert OpenCV image to PIL
            if len(image.shape) == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
            
            # Use Python tesseract package
            result = tesseract.image_to_data(image_pil, output_type=tesseract.Output.DICT)
            return result
        except (ImportError, SyntaxError) as e:
            logger.error(f"Python tesseract package has syntax error or import issue: {e}")
            # Return empty result structure
            return {
                "level": [], "left": [], "top": [], "width": [], "height": [],
                "text": [], "conf": []
            }
        except Exception as e:
            logger.error(f"Python tesseract fallback failed: {e}")
            # Return empty result structure
            return {
                "level": [], "left": [], "top": [], "width": [], "height": [],
                "text": [], "conf": []
            }
    
    def _run_python_tesseract_ocr_fallback(self, image) -> dict:
        """Fallback method using Python tesseract_ocr package."""
        try:
            import tesseract_ocr
            # Convert image to format expected by tesseract_ocr
            import numpy as np
            from PIL import Image
            
            # Convert OpenCV image to PIL
            if len(image.shape) == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
            
            # Use tesseract_ocr package
            result = tesseract_ocr.image_to_data(image_pil, output_type=tesseract_ocr.Output.DICT)
            return result
        except (ImportError, SyntaxError) as e:
            logger.error(f"Python tesseract_ocr package has syntax error or import issue: {e}")
            # Return empty result structure
            return {
                "level": [], "left": [], "top": [], "width": [], "height": [],
                "text": [], "conf": []
            }
        except Exception as e:
            logger.error(f"Python tesseract_ocr fallback failed: {e}")
            # Return empty result structure
            return {
                "level": [], "left": [], "top": [], "width": [], "height": [],
                "text": [], "conf": []
            } 