from __future__ import annotations

"""Grounding engine using UI-Venus model (inclusionAI/UI-Venus-Ground-7B).

The engine feeds a screenshot to the model with a concise prompt asking for a
JSON list of UI elements (text + bounding box). It then parses the JSON and
returns ``UIElement`` objects that can be consumed by the rest of the
framework.

NOTE:  This first cut keeps things simple: it runs the model in *text* mode, so
we rely on the model’s built-in image tokenizer (similar to Phi-Ground).  No OCR
or template matching is used.
"""

from typing import Any, List, Dict
import json
import os

from PIL import Image
from loguru import logger
import torch

# Prefer the dedicated Qwen2 vision-language class when present (needed for
# inclusionAI/UI-Venus-Ground-7B and other Qwen-VL derivatives).  Fallback to the
# generic AutoModelForCausalLM so the module still works on older Transformers
# versions.

try:  # Transformers ≥ 4.41.0
    from transformers import Qwen2VLForConditionalGeneration  # type: ignore
    _QWEN_VL_AVAILABLE = True
except ImportError:  # Older transformers
    from transformers import AutoModelForCausalLM  # type: ignore
    Qwen2VLForConditionalGeneration = None  # placeholder
    _QWEN_VL_AVAILABLE = False

from transformers import AutoProcessor

from ..core.config import config
from .models import UIElement, BoundingBox


class UIVenusGroundingEngine:  # noqa: D101
    def __init__(self, model_name: str | None = None) -> None:  # noqa: D401,E501
        self.model_name = model_name or config.ui_venus_model
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading UI-Venus grounding model on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, token=self.hf_token
        )

        model_cls = (
            Qwen2VLForConditionalGeneration if _QWEN_VL_AVAILABLE else AutoModelForCausalLM
        )

        self.model = model_cls.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            token=self.hf_token,
            low_cpu_mem_usage=True,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def detect_ui_elements(self, screenshot: str) -> List[UIElement]:  # noqa: D401,E501
        """Return UI elements detected in *screenshot* using UI-Venus."""
        if not os.path.exists(screenshot):
            logger.error(f"Screenshot not found: {screenshot}")
            return []

        image = Image.open(screenshot).convert("RGB")

        prompt = (
            "<|system|>You are a mobile-UI grounding agent.  "
            "List ALL clickable or text input elements visible in the provided "
            "Android screenshot as JSON array.  Each item must have: 'text', "
            "'bbox' (left,top,right,bottom integers).  Respond with JSON only, "
            "no explanation.<|end|>"
        )

        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        output = self.processor.batch_decode(generated, skip_special_tokens=True)[0]

        logger.debug(f"UI-Venus raw grounding output: {output[:300]}")

        # Attempt to locate JSON in output
        json_start = output.find("[")
        json_end = output.rfind("]")
        if json_start == -1 or json_end == -1:
            logger.error("Could not find JSON list in Venus response")
            return []
        try:
            elements_data: List[Dict[str, Any]] = json.loads(output[json_start : json_end + 1])
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse Venus JSON: {exc}")
            return []

        elements: List[UIElement] = []
        for item in elements_data:
            try:
                text = str(item.get("text", "")).strip()
                bbox_vals = item.get("bbox") or item.get("bounding_box")
                if not bbox_vals or len(bbox_vals) != 4:
                    continue
                left, top, right, bottom = map(int, bbox_vals)
                bbox = BoundingBox(left, top, right, bottom)
                elements.append(UIElement(bbox=bbox, text=text, confidence=1.0))
            except Exception as parsed_exc:  # noqa: BLE001
                logger.debug(f"Skip bad item {item}: {parsed_exc}")

        logger.info(f"UI-Venus grounding detected {len(elements)} elements")
        return elements


# Singleton accessor ---------------------------------------------------------

_ui_venus_grounder: UIVenusGroundingEngine | None = None


def get_ui_venus_grounder() -> UIVenusGroundingEngine:  # noqa: D401
    global _ui_venus_grounder  # pylint: disable=global-statement
    if _ui_venus_grounder is None:
        _ui_venus_grounder = UIVenusGroundingEngine()
    return _ui_venus_grounder
