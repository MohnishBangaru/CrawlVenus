"""Explorer: thin alias module.

This module provides the *Explorer* class by re-exporting the existing
``DroidBotGPT`` implementation.  It lets us migrate imports from
``core.explorer_gpt`` to ``core.explorer`` without touching the 1,000-line
implementation file yet.  A full class rename can follow later if desired.
"""

from __future__ import annotations

from .explorer_gpt import DroidBotGPT as Explorer  # noqa: F401

__all__: list[str] = ["Explorer"] 