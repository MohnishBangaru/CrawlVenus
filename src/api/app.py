"""FastAPI application factory for DroidBot-GPT framework."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import config
from ..core.logger import log
from .routes import automation_router, device_router, status_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="DroidBot-GPT API",
        description="Intelligent Android automation framework API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        log.info(f"API Request: {request.method} {request.url}")
        response = await call_next(request)
        log.info(f"API Response: {response.status_code}")
        return response
    
    # Include routers
    app.include_router(device_router, prefix="/api/v1/device", tags=["device"])
    app.include_router(automation_router, prefix="/api/v1/automation", tags=["automation"])
    app.include_router(status_router, prefix="/api/v1/status", tags=["status"])
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "DroidBot-GPT API",
            "version": "1.0.0"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "DroidBot-GPT API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    log.info("FastAPI application created successfully")
    return app 