"""Async OpenAI client wrapper with retry & singleton semantics."""
from __future__ import annotations

import asyncio
import random
from typing import Any

import openai  # type: ignore

from src.core.config import config
from src.core.logger import log

__all__ = ["OpenAIClient", "get_openai_client"]

# Constant settings
_MAX_RETRIES = 4
_BASE_BACKOFF = 1.0  # seconds

# Vision-capable models
VISION_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini"
]

# Fallback model for non-vision requests
DEFAULT_MODEL = "gpt-4"


class OpenAIClient:
    """Lightweight async wrapper around OpenAI chat completion API."""

    _instance: OpenAIClient | None = None

    @classmethod
    def instance(cls) -> OpenAIClient:
        """Return the singleton instance, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the underlying async OpenAI client (internal use)."""
        # Ensure singleton creation only once
        if OpenAIClient._instance is not None:
            raise RuntimeError("Use OpenAIClient.instance() instead of constructor")

        api_key = config.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        # Use new OpenAI 1.x client (sync).  Async version via openai.AsyncOpenAI
        self._client = openai.AsyncOpenAI(api_key=api_key)

        self.model = config.openai_model
        self.temperature = float(config.openai_temperature)
        self.max_tokens = int(config.openai_max_tokens)

    def _has_image_content(self, messages: list[dict[str, Any]]) -> bool:
        """Check if messages contain image content."""
        for message in messages:
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        return True
        return False

    def _get_appropriate_model(self, messages: list[dict[str, Any]]) -> str:
        """Get the appropriate model based on message content."""
        if self._has_image_content(messages):
            # Use vision-capable model
            if self.model in VISION_MODELS:
                return self.model
            else:
                # Use a vision-capable model as fallback
                vision_model = "gpt-4o"  # Most recent vision model
                log.info(f"Switching to vision-capable model: {vision_model}")
                return vision_model
        else:
            # Use configured model for text-only requests
            return self.model

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def chat(
        self,
        *,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Send chat completion request and return assistant reply.

        Args:
            prompt: Convenience user prompt string. Ignored if ``messages`` is provided.
            system_prompt: System prompt string (used if ``messages`` is ``None``).
            messages: Full message list to pass through; takes precedence over *prompt*.

        """
        if messages is None:
            if prompt is None:
                raise ValueError("Either `messages` or `prompt` must be provided")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        # Otherwise: caller supplied full chat history via *messages*.

        backoff = _BASE_BACKOFF
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content  # type: ignore[attr-defined]
                if content is None:
                    raise RuntimeError("OpenAI returned empty content")
                return content
            except (openai.APIError, openai.RateLimitError) as exc:
                if attempt == _MAX_RETRIES - 1:
                    log.error(f"OpenAI request failed after {attempt+1} attempts: {exc}")
                    raise
                sleep_time = backoff * (2 ** attempt) + random.uniform(0, 0.5)  # noqa: S311
                log.warning(f"OpenAI error {exc}. Retrying in {sleep_time:.1f}s…")
                await asyncio.sleep(sleep_time)
                continue

        # Should not reach here
        raise RuntimeError("OpenAI chat completion failed after retries")

    async def get_completion(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send chat completion request with support for text and image messages.

        Args:
            messages: List of message dictionaries that can include text and image content.

        Returns:
            Parsed response as dictionary
        """
        # Determine appropriate model based on content
        model = self._get_appropriate_model(messages)
        
        backoff = _BASE_BACKOFF
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content  # type: ignore[attr-defined]
                if content is None:
                    raise RuntimeError("OpenAI returned empty content")
                
                # Try to parse as JSON
                try:
                    import json
                    # Find JSON in the response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        return json.loads(json_str)
                    else:
                        # If no JSON found, return as text
                        return {"analysis": {"reasoning": content}}
                except json.JSONDecodeError:
                    # If JSON parsing fails, return as text
                    return {"analysis": {"reasoning": content}}
                    
            except (openai.APIError, openai.RateLimitError) as exc:
                if attempt == _MAX_RETRIES - 1:
                    log.error(f"OpenAI request failed after {attempt+1} attempts: {exc}")
                    raise
                sleep_time = backoff * (2 ** attempt) + random.uniform(0, 0.5)  # noqa: S311
                log.warning(f"OpenAI error {exc}. Retrying in {sleep_time:.1f}s…")
                await asyncio.sleep(sleep_time)
                continue

        # Should not reach here
        raise RuntimeError("OpenAI chat completion failed after retries")


# Convenience getter
get_openai_client = OpenAIClient.instance 