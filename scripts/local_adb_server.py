#!/usr/bin/env python3
"""
Local ADB Server for Distributed AA_VA-Phi Setup
================================================

This script runs on your laptop and provides a REST API for the RunPod instance
to interact with your local Android emulator via ADB.

Usage:
    python scripts/local_adb_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import json
import logging
import subprocess
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local ADB Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ScreenshotRequest(BaseModel):
    output_path: Optional[str] = None
    quality: int = 90

class TapRequest(BaseModel):
    x: int
    y: int

class SwipeRequest(BaseModel):
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration: int = 500

class KeyEventRequest(BaseModel):
    key_code: int

class InstallRequest(BaseModel):
    apk_path: str

class DeviceInfo(BaseModel):
    device_id: str
    status: str
    platform_version: Optional[str] = None

class ADBResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: float = time.time()


def run_adb_command(command: List[str], timeout: int = 30, binary: bool = False) -> Dict[str, Any]:
    """Run ADB command and return result."""
    try:
        logger.info(f"Running ADB command: {' '.join(command)}")
        result = subprocess.run(
            ['adb'] + command,
            capture_output=True,
            text=not binary,  # Use binary mode for screenshots
            timeout=timeout
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Command timed out after {timeout} seconds',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'returncode': -1
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/devices", response_model=List[DeviceInfo])
async def get_devices():
    """Get connected devices."""
    result = run_adb_command(['devices'])
    if not result['success']:
        raise HTTPException(status_code=500, detail=f"Failed to get devices: {result['error']}")
    
    devices = []
    for line in result['stdout'].split('\n')[1:]:  # Skip header
        if line.strip():
            parts = line.split('\t')
            if len(parts) >= 2:
                devices.append(DeviceInfo(
                    device_id=parts[0],
                    status=parts[1]
                ))
    
    return devices


@app.post("/screenshot", response_model=ADBResponse)
async def take_screenshot(request: ScreenshotRequest):
    """Take screenshot from device."""
    output_path = request.output_path or f"screenshot_{int(time.time())}.png"
    
    # Use fast screencap method with binary output
    result = run_adb_command(['exec-out', 'screencap -p'], timeout=60, binary=True)
    
    if result['success']:
        try:
            # Save screenshot locally
            with open(output_path, 'wb') as f:
                f.write(result['stdout'])
            
            logger.info(f"Screenshot saved to file: {output_path}, size: {len(result['stdout'])} bytes")
            
            # Return both file path and base64 data (compressed for transmission)
            import base64
            import gzip
            
            # Compress the screenshot data to reduce transmission size
            compressed_data = gzip.compress(result['stdout'])
            screenshot_data = base64.b64encode(compressed_data).decode('utf-8')
            logger.info(f"Screenshot compressed: {len(result['stdout'])} -> {len(screenshot_data)} bytes")
            
            response_data = {
                'output_path': output_path, 
                'size': len(result['stdout']),
                'screenshot_data': screenshot_data,
                'compressed': True
            }
            
            logger.info(f"Screenshot saved: {output_path} ({len(result['stdout'])} bytes)")
            return ADBResponse(success=True, data=response_data)
        except Exception as e:
            logger.error(f"Screenshot save failed: {e}")
            return ADBResponse(success=False, error=str(e))
    else:
        logger.error(f"Screenshot capture failed: {result['error']}")
        return ADBResponse(success=False, error=result['error'])


@app.post("/tap", response_model=ADBResponse)
async def tap_screen(request: TapRequest):
    """Tap on screen coordinates."""
    result = run_adb_command(['shell', 'input', 'tap', str(request.x), str(request.y)])
    
    if result['success']:
        return ADBResponse(success=True, data={'x': request.x, 'y': request.y})
    else:
        return ADBResponse(success=False, error=result['error'])


@app.post("/swipe", response_model=ADBResponse)
async def swipe_screen(request: SwipeRequest):
    """Swipe on screen."""
    result = run_adb_command([
        'shell', 'input', 'swipe',
        str(request.start_x), str(request.start_y),
        str(request.end_x), str(request.end_y),
        str(request.duration)
    ])
    
    if result['success']:
        return ADBResponse(success=True, data={
            'start_x': request.start_x,
            'start_y': request.start_y,
            'end_x': request.end_x,
            'end_y': request.end_y,
            'duration': request.duration
        })
    else:
        return ADBResponse(success=False, error=result['error'])


@app.post("/keyevent", response_model=ADBResponse)
async def send_keyevent(request: KeyEventRequest):
    """Send key event."""
    result = run_adb_command(['shell', 'input', 'keyevent', str(request.key_code)])
    
    if result['success']:
        return ADBResponse(success=True, data={'key_code': request.key_code})
    else:
        return ADBResponse(success=False, error=result['error'])


@app.post("/install", response_model=ADBResponse)
async def install_apk(request: InstallRequest):
    """Install APK file."""
    if not Path(request.apk_path).exists():
        return ADBResponse(success=False, error=f"APK file not found: {request.apk_path}")
    
    result = run_adb_command(['install', '-r', request.apk_path], timeout=120)
    
    if result['success']:
        return ADBResponse(success=True, data={'apk_path': request.apk_path})
    else:
        return ADBResponse(success=False, error=result['error'])


@app.post("/uninstall", response_model=ADBResponse)
async def uninstall_app(package_name: str):
    """Uninstall app by package name."""
    result = run_adb_command(['uninstall', package_name])
    
    if result['success']:
        return ADBResponse(success=True, data={'package_name': package_name})
    else:
        return ADBResponse(success=False, error=result['error'])


@app.post("/launch", response_model=ADBResponse)
async def launch_app(package_name: str, activity: str = None):
    """Launch app."""
    if activity:
        command = ['shell', 'am', 'start', '-n', f'{package_name}/{activity}']
    else:
        command = ['shell', 'monkey', '-p', package_name, '-c', 'android.intent.category.LAUNCHER', '1']
    
    result = run_adb_command(command)
    
    if result['success']:
        return ADBResponse(success=True, data={'package_name': package_name, 'activity': activity})
    else:
        return ADBResponse(success=False, error=result['error'])


@app.get("/foreground", response_model=ADBResponse)
async def get_foreground_app():
    """Get current foreground app."""
    result = run_adb_command(['shell', 'dumpsys', 'activity', 'activities', '|', 'grep', 'mResumedActivity'])
    
    if result['success']:
        return ADBResponse(success=True, data={'foreground_app': result['stdout']})
    else:
        return ADBResponse(success=False, error=result['error'])


@app.post("/shell", response_model=ADBResponse)
async def execute_shell_command(request: dict):
    """Execute shell command on device."""
    command = request.get('command', '')
    if not command:
        return ADBResponse(success=False, error="No command provided")
    
    result = run_adb_command(['shell'] + command.split())
    
    if result['success']:
        return ADBResponse(success=True, data={'output': result['stdout']})
    else:
        return ADBResponse(success=False, error=result['error'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local ADB Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Local ADB Server on {args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info"
    )
