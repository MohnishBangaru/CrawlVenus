"""Distributed configuration for RunPod + Local Emulator setup."""

import os
import socket
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class DistributedConfig(BaseSettings):
    """Configuration for distributed RunPod + Local Emulator setup."""
    
    # Distributed Mode
    distributed_mode: bool = Field(default=False, description="Enable distributed mode")
    
    # Local ADB Configuration
    local_adb_host: str = Field(default="localhost", description="Local laptop IP address")
    local_adb_port: int = Field(default=5037, description="Local ADB port")
    local_adb_timeout: int = Field(default=30, description="ADB connection timeout")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # WebSocket Configuration
    websocket_host: str = Field(default="0.0.0.0", description="WebSocket server host")
    websocket_port: int = Field(default=8001, description="WebSocket server port")
    
    # Connection Settings
    connection_retries: int = Field(default=3, description="Number of connection retries")
    connection_timeout: int = Field(default=10, description="Connection timeout in seconds")
    
    # Security
    enable_ssl: bool = Field(default=False, description="Enable SSL for API")
    ssl_cert_file: Optional[str] = Field(default=None, description="SSL certificate file")
    ssl_key_file: Optional[str] = Field(default=None, description="SSL private key file")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    enable_remote_logging: bool = Field(default=True, description="Enable remote logging")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore unexpected env vars rather than raising errors
    
    def get_local_adb_url(self) -> str:
        """Get local ADB connection URL."""
        return f"{self.local_adb_host}:{self.local_adb_port}"
    
    def get_api_url(self) -> str:
        """Get API server URL."""
        protocol = "https" if self.enable_ssl else "http"
        return f"{protocol}://{self.api_host}:{self.api_port}"
    
    def get_websocket_url(self) -> str:
        """Get WebSocket server URL."""
        protocol = "wss" if self.enable_ssl else "ws"
        return f"{protocol}://{self.websocket_host}:{self.websocket_port}"
    
    def validate_network_connectivity(self) -> bool:
        """Validate network connectivity to local ADB."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            result = sock.connect_ex((self.local_adb_host, self.local_adb_port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"Network connectivity check failed: {e}")
            return False


# Global distributed config instance
distributed_config = DistributedConfig()
