"""Parse and validate LLM critique response JSON."""

import json
import re
from dataclasses import dataclass, field

__all__ = ["Critique", "parse_critique"]

_REQUIRED_KEYS = {
    "strengths": list,
    "issues": list,
    "uniqueness_vs_generic": list,
    "summary_50_words": str,
}


@dataclass
class Critique:
    """Structured representation of an LLM critique response."""

    strengths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    uniqueness_vs_generic: list[str] = field(default_factory=list)
    summary_50_words: str = ""
    summary_word_count: int = 0
    valid: bool = False
    errors: list[str] = field(default_factory=list)


_JSON_REGEX = re.compile(r"\{[\s\S]+\}")


def _first_json_blob(text: str) -> str | None:
    """Return first JSON-looking {...} block from text."""
    match = _JSON_REGEX.search(text)
    return match.group(0) if match else None


def _count_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s))


def parse_critique(raw: str) -> Critique:
    """Attempt to parse LLM critique JSON and validate summary word count."""
    result = Critique()

    json_str = _first_json_blob(raw)
    if not json_str:
        result.errors.append("No JSON object found in response")
        return result

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        result.errors.append(f"JSON decode error: {exc}")
        return result

    # Validate required keys & types
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in data:
            result.errors.append(f"Missing key: {key}")
            continue
        if not isinstance(data[key], expected_type):
            result.errors.append(f"Key {key} expected {expected_type.__name__}")

    # Populate fields
    result.strengths = data.get("strengths", [])
    result.issues = data.get("issues", [])
    result.uniqueness_vs_generic = data.get("uniqueness_vs_generic", [])
    result.summary_50_words = data.get("summary_50_words", "")
    result.summary_word_count = _count_words(result.summary_50_words)

    if result.summary_word_count != 50:
        result.errors.append(
            f"summary_50_words has {result.summary_word_count} words (expected 50)"
        )

    result.valid = not result.errors
    return result 