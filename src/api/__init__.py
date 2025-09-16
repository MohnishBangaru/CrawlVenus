"""FastAPI endpoints for DroidBot-GPT framework.

This sub-package provides REST API endpoints for:
- Device management and control
- Automation task execution
- Status monitoring and reporting
- Configuration management
"""

from .app import create_app
from .routes import automation_router, device_router, status_router

__all__ = [
    "create_app",
    "automation_router", 
    "device_router",
    "status_router",
] 