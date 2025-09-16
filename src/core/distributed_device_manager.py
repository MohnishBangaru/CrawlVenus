"""Distributed Device Manager for RunPod + Local Emulator Setup."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import aiohttp
import requests
from PIL import Image
import io

from .distributed_config import distributed_config

logger = logging.getLogger(__name__)


class DistributedDeviceManager:
    """Device manager that communicates with local ADB server."""
    
    def __init__(self, local_server_url: str = None):
        """Initialize distributed device manager."""
        self.local_server_url = local_server_url or f"http://{distributed_config.local_adb_host}:{distributed_config.local_adb_port}"
        self.session = None
        self.device_id = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to local ADB server."""
        url = f"{self.local_server_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, **kwargs)
            elif method.upper() == "POST":
                # Extract json data and remove from kwargs
                json_data = kwargs.pop('json', None)
                response = requests.post(url, json=json_data, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _make_async_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make async HTTP request to local ADB server."""
        url = f"{self.local_server_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method.upper() == "POST":
                async with self.session.post(url, json=kwargs.get('json'), **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            logger.error(f"Async request failed: {e}")
            return {"success": False, "error": str(e)}
    
    def check_connectivity(self) -> bool:
        """Check connectivity to local ADB server."""
        try:
            response = self._make_request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            return False
    
    def get_devices(self) -> List[Dict[str, str]]:
        """Get connected devices from local ADB server."""
        response = self._make_request("GET", "/devices")
        # Handle direct list response from FastAPI
        if isinstance(response, list):
            return response
        elif response.get("success", False):
            return response.get("data", [])
        else:
            logger.error(f"Failed to get devices: {response.get('error')}")
            return []
    
    def take_screenshot(self, output_path: str = None) -> Optional[bytes]:
        """Take screenshot from device via local ADB server."""
        request_data = {}
        if output_path:
            request_data["output_path"] = output_path
        
        response = self._make_request("POST", "/screenshot", json=request_data)
        
        if response.get("success", False):
            # Return screenshot data as bytes
            screenshot_data = response.get("data", {}).get("screenshot_data")
            if screenshot_data:
                import base64
                import gzip
                
                decoded_data = base64.b64decode(screenshot_data)
                
                # Check if data is compressed
                if response.get("data", {}).get("compressed", False):
                    decoded_data = gzip.decompress(decoded_data)
                
                return decoded_data
            else:
                # If no base64 data, try to read the file and return bytes
                file_path = response.get("data", {}).get("output_path")
                if file_path and Path(file_path).exists():
                    try:
                        with open(file_path, 'rb') as f:
                            data = f.read()
                            return data
                    except Exception as e:
                        logger.error(f"Failed to read screenshot file {file_path}: {e}")
                        return None
                else:
                    logger.error(f"No screenshot data or file path available")
                    return None
        else:
            logger.error(f"Failed to take screenshot: {response.get('error')}")
            return None
    
    def tap(self, x: int, y: int) -> bool:
        """Tap on screen coordinates via local ADB server."""
        request_data = {"x": x, "y": y}
        response = self._make_request("POST", "/tap", json=request_data)
        
        if response.get("success", False):
            logger.info(f"Successfully tapped at ({x}, {y})")
            return True
        else:
            logger.error(f"Failed to tap at ({x}, {y}): {response.get('error')}")
            return False
    
    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 500) -> bool:
        """Swipe on screen via local ADB server."""
        request_data = {
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "duration": duration
        }
        response = self._make_request("POST", "/swipe", json=request_data)
        
        if response.get("success", False):
            logger.info(f"Successfully swiped from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True
        else:
            logger.error(f"Failed to swipe: {response.get('error')}")
            return False
    
    def send_keyevent(self, key_code: int) -> bool:
        """Send key event via local ADB server."""
        request_data = {"key_code": key_code}
        response = self._make_request("POST", "/keyevent", json=request_data)
        
        if response.get("success", False):
            logger.info(f"Successfully sent key event: {key_code}")
            return True
        else:
            logger.error(f"Failed to send key event {key_code}: {response.get('error')}")
            return False
    
    def install_apk(self, apk_path: str) -> bool:
        """Install APK via local ADB server."""
        request_data = {"apk_path": apk_path}
        response = self._make_request("POST", "/install", json=request_data)
        
        if response.get("success", False):
            logger.info(f"Successfully installed APK: {apk_path}")
            return True
        else:
            logger.error(f"Failed to install APK {apk_path}: {response.get('error')}")
            return False
    
    def uninstall_app(self, package_name: str) -> bool:
        """Uninstall app via local ADB server."""
        response = self._make_request("POST", f"/uninstall?package_name={package_name}")
        
        if response.get("success", False):
            logger.info(f"Successfully uninstalled app: {package_name}")
            return True
        else:
            logger.error(f"Failed to uninstall app {package_name}: {response.get('error')}")
            return False
    
    def launch_app(self, package_name: str, activity: str = None) -> bool:
        """Launch app via local ADB server."""
        params = {"package_name": package_name}
        if activity:
            params["activity"] = activity
        
        response = self._make_request("POST", "/launch", params=params)
        
        if response.get("success", False):
            logger.info(f"Successfully launched app: {package_name}")
            return True
        else:
            logger.error(f"Failed to launch app {package_name}: {response.get('error')}")
            return False
    
    def get_foreground_app(self) -> Optional[str]:
        """Get current foreground app via local ADB server."""
        response = self._make_request("GET", "/foreground")
        
        if response.get("success", False):
            return response.get("data", {}).get("foreground_app")
        else:
            logger.error(f"Failed to get foreground app: {response.get('error')}")
            return None
    
    def execute_shell_command(self, command: str) -> Optional[str]:
        """Execute shell command via local ADB server."""
        request_data = {"command": command}
        response = self._make_request("POST", "/shell", json=request_data)
        
        if response.get("success", False):
            return response.get("data", {}).get("output")
        else:
            logger.error(f"Failed to execute shell command '{command}': {response.get('error')}")
            return None
    
    def wait_for_device(self, timeout: int = 30) -> bool:
        """Wait for device to be available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_connectivity():
                devices = self.get_devices()
                if devices and any(d.get("status") == "device" for d in devices):
                    logger.info("Device is available")
                    return True
            time.sleep(1)
        
        logger.error("Device not available within timeout")
        return False
    
    def optimize_device_settings(self) -> bool:
        """Optimize device settings for automation."""
        try:
            # Disable animations
            self.execute_shell_command("settings put global window_animation_scale 0")
            self.execute_shell_command("settings put global transition_animation_scale 0")
            self.execute_shell_command("settings put global animator_duration_scale 0")
            
            # Enable developer options
            self.execute_shell_command("settings put global development_settings_enabled 1")
            
            # Set screen timeout to longer duration
            self.execute_shell_command("settings put system screen_off_timeout 300000")
            
            logger.info("Device settings optimized for automation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize device settings: {e}")
            return False


# Convenience functions for easy usage
def create_distributed_device_manager(local_server_url: str = None) -> DistributedDeviceManager:
    """Create a distributed device manager instance."""
    return DistributedDeviceManager(local_server_url)


async def test_distributed_connection(local_server_url: str = None) -> bool:
    """Test connection to local ADB server."""
    async with DistributedDeviceManager(local_server_url) as manager:
        if manager.check_connectivity():
            devices = manager.get_devices()
            logger.info(f"Connected to local ADB server. Found {len(devices)} devices.")
            return True
        else:
            logger.error("Failed to connect to local ADB server")
            return False
