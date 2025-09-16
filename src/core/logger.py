"""DroidNot structured logging system."""

from __future__ import annotations

import os
import sys
from typing import Any

from loguru import logger

from .config import config


class Logger:
    """Structured logging system for DroidBot-GPT framework."""
    
    def __init__(self, name: str = "DroidNot") -> None:
        """Initialize and configure a *Loguru* logger instance."""
        self.name = name
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Configure the logger with proper formatting and handlers."""
        # Remove default handler
        logger.remove()
        
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # ------------------------------------------------------------------
        # Console handler
        # ------------------------------------------------------------------
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stdout,
            format=console_format,
            level=config.log_level,
            colorize=True,
        )
        
        # ------------------------------------------------------------------
        # File handlers
        # ------------------------------------------------------------------
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
            "{name}:{function}:{line} | {message}"
        )

        logger.add(
            os.path.join(logs_dir, "droidbot_{time:YYYY-MM-DD}.log"),
            format=file_format,
            level="DEBUG",
            rotation="1 day",
            retention="30 days",
            compression="zip",
        )
        
        # Separate error log
        logger.add(
            os.path.join(logs_dir, "errors_{time:YYYY-MM-DD}.log"),
            format=file_format,
            level="ERROR",
            rotation="1 day",
            retention="90 days",
            compression="zip",
        )
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        logger.info(f"[{self.name}] {message}", **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        logger.debug(f"[{self.name}] {message}", **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        logger.warning(f"[{self.name}] {message}", **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        logger.error(f"[{self.name}] {message}", **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        logger.critical(f"[{self.name}] {message}", **kwargs)
    
    def success(self, message: str, **kwargs: Any) -> None:
        """Log success message."""
        logger.success(f"[{self.name}] {message}", **kwargs)
    
    def log_automation_step(self, step: str, details: dict[str, Any] | None = None) -> None:
        """Log automation step with details."""
        message = f"AUTOMATION STEP: {step}"
        if details:
            message += f" | Details: {details}"
        self.info(message)
    
    def log_ai_decision(
        self,
        decision: str,
        confidence: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log AI decision with confidence and context."""
        msg = f"AI DECISION: {decision} (confidence: {confidence:.2f})"
        if context:
            msg += f" | Context: {context}"
        self.info(msg)
    
    def log_vision_detection(
        self,
        element_type: str,
        confidence: float,
        coordinates: tuple[int, int],
    ) -> None:
        """Log computer vision detection results."""
        msg = (
            f"VISION DETECTION: {element_type} at {coordinates} "
            f"(confidence: {confidence:.2f})"
        )
        self.debug(msg)
    
    def log_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Log resource usage metrics."""
        self.debug(f"RESOURCE USAGE: Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
    
    def log_performance(self, operation: str, duration_ms: float) -> None:
        """Log performance metrics."""
        self.debug(f"PERFORMANCE: {operation} took {duration_ms:.2f}ms")


# Global logger instance
log = Logger() 