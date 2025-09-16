"""Prompt construction utilities for Explorer (the GPT-driven Android tester).

Phase-3 Step-1: Build a compact, structured prompt where the LLM acts as a
Human App Critic, analysing UI/UX, standout features, and generic bits.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..vision.models import UIElement

HEADER = (
    "You are a seasoned Human App Critic. You analyse mobile apps from both "
    "a user-experience and functionality standpoint. At each screen you will "
    "receive: 1) a list of visible UI elements with bounding-box positions, "
    "2) the task the automation is trying to complete, and 3) a short action "
    "history. Your goals:\n"
    "  • Identify notable UI / UX details that stand out (good or bad).\n"
    "  • Note the BEST features offered on this screen.\n"
    "  • Point out features that feel GENERIC or uninteresting.\n"
    "  • Decide the next UI action that moves closer to completing the task.\n"
    "Return a JSON object with keys:\n"
    "    critic_notes: string   # your analysis in <120 words\n"
    "    next_action:          # JSON describing the UI action\n"
    "        type: tap|swipe|input_text|wait|key_event\n"
    "        ... parameters depending on type\n"
    "    confidence: float     # 0-1 how sure you are about next_action\n"
)


def _serialize_elements(elements: list[UIElement], max_elems: int = 20) -> str:
    """Convert UIElement list to a concise multiline string."""
    lines: list[str] = []
    for idx, el in enumerate(elements[:max_elems], 1):
        x1, y1, x2, y2 = el.bbox.as_tuple()
        bbox = f"({x1},{y1})-({x2},{y2})"
        line = f"{idx}. [{el.element_type}] {el.text!r} {bbox} conf={el.confidence:.2f}"
        lines.append(line)
    if len(elements) > max_elems:
        lines.append(f"… and {len(elements) - max_elems} more elements omitted")
    return "\n".join(lines)


def build_prompt(
    state: dict[str, Any],
    task: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Return full prompt string for LLM call."""
    elements: list[UIElement] = state.get("ui_elements", [])  # type: ignore[assignment]
    elements_str = _serialize_elements(elements)

    hist_lines: list[str] = []
    if history:
        for h in history[-5:]:  # last 5 actions
            hist_lines.append(f"• {h['action']['type']} => {h['result']['success']}")
    history_str = "\n".join(hist_lines) if hist_lines else "(no previous actions)"

    prompt = (
        f"{HEADER}\n"
        f"=== Current Task ===\n{task}\n"
        f"=== Visible Elements ===\n{elements_str}\n"
        f"=== Recent Actions ===\n{history_str}\n"
        f"### JSON RESPONSE ONLY ###"
    )

    logger.debug("Prompt generated, {0} characters", len(prompt))
    return prompt 