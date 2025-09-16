"""API route definitions for DroidBot-GPT framework."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.explorer import Explorer
from ..core.logger import log

# Create router instances
device_router = APIRouter()
automation_router = APIRouter()
status_router = APIRouter()

# Global DroidBot-GPT instance
droidbot_instance: Optional[Explorer] = None


def get_droidbot() -> Explorer:
    """Get or create the global DroidBot-GPT instance."""
    global droidbot_instance
    if droidbot_instance is None:
        droidbot_instance = Explorer()
    return droidbot_instance


# Pydantic models for request/response
class TaskRequest(BaseModel):
    """Request model for automation tasks."""
    description: str
    max_steps: int = 50


class ActionRequest(BaseModel):
    """Request model for individual actions."""
    action_type: str
    parameters: Dict[str, Any]


class DeviceConnectRequest(BaseModel):
    """Request model for device connection."""
    device_serial: Optional[str] = None


class TaskResponse(BaseModel):
    """Response model for task execution."""
    task_id: str
    status: str
    duration: float
    steps: int
    result: Dict[str, Any]


class DeviceInfoResponse(BaseModel):
    """Response model for device information."""
    connected: bool
    device_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Device management routes
@device_router.post("/connect", response_model=DeviceInfoResponse)
async def connect_device(request: DeviceConnectRequest):
    """Connect to an Android device or emulator."""
    try:
        droidbot = get_droidbot()
        success = await droidbot.connect_device(request.device_serial)
        
        if success:
            device_status = await droidbot.get_device_status()
            return DeviceInfoResponse(
                connected=True,
                device_info=device_status.get('device_info')
            )
        else:
            return DeviceInfoResponse(
                connected=False,
                error="Failed to connect to device"
            )
            
    except Exception as e:
        log.error(f"Device connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@device_router.get("/status", response_model=DeviceInfoResponse)
async def get_device_status():
    """Get current device status."""
    try:
        droidbot = get_droidbot()
        device_status = await droidbot.get_device_status()
        
        return DeviceInfoResponse(
            connected=device_status.get('connected', False),
            device_info=device_status.get('device_info')
        )
        
    except Exception as e:
        log.error(f"Device status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@device_router.post("/disconnect")
async def disconnect_device():
    """Disconnect from the current device."""
    try:
        droidbot = get_droidbot()
        await droidbot.disconnect()
        return {"message": "Device disconnected successfully"}
        
    except Exception as e:
        log.error(f"Device disconnection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@device_router.post("/screenshot")
async def capture_screenshot():
    """Capture a screenshot from the device."""
    try:
        droidbot = get_droidbot()
        device_manager = droidbot.device_manager
        
        if not device_manager.is_connected():
            raise HTTPException(status_code=400, detail="Device not connected")
            
        screenshot_path = await device_manager.capture_screenshot()
        
        return {
            "screenshot_path": screenshot_path,
            "message": "Screenshot captured successfully"
        }
        
    except Exception as e:
        log.error(f"Screenshot capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Automation routes
@automation_router.post("/task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute an automation task."""
    try:
        droidbot = get_droidbot()
        
        if not droidbot.device_manager.is_connected():
            raise HTTPException(status_code=400, detail="Device not connected")
            
        result = await droidbot.automate_task(
            request.description,
            request.max_steps
        )
        
        return TaskResponse(
            task_id=result.get('id', 'unknown'),
            status=result.get('status', 'unknown'),
            duration=result.get('duration', 0.0),
            steps=result.get('total_steps', 0),
            result=result
        )
        
    except Exception as e:
        log.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@automation_router.get("/tasks", response_model=List[Dict[str, Any]])
async def get_task_history():
    """Get automation task history."""
    try:
        droidbot = get_droidbot()
        history = await droidbot.get_task_history()
        return history
        
    except Exception as e:
        log.error(f"Task history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@automation_router.get("/current-task")
async def get_current_task():
    """Get information about the currently running task."""
    try:
        droidbot = get_droidbot()
        current_task = await droidbot.get_current_task()
        return current_task
        
    except Exception as e:
        log.error(f"Current task error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@automation_router.post("/stop")
async def stop_automation():
    """Stop the currently running automation."""
    try:
        droidbot = get_droidbot()
        await droidbot.stop_automation()
        return {"message": "Automation stopped successfully"}
        
    except Exception as e:
        log.error(f"Stop automation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@automation_router.post("/action")
async def execute_action(request: ActionRequest):
    """Execute a single automation action."""
    try:
        droidbot = get_droidbot()
        
        if not droidbot.device_manager.is_connected():
            raise HTTPException(status_code=400, detail="Device not connected")
            
        action = {
            "type": request.action_type,
            **request.parameters
        }
        
        result = await droidbot.device_manager.perform_action(action)
        
        return {
            "success": result,
            "action": action,
            "message": "Action executed successfully" if result else "Action failed"
        }
        
    except Exception as e:
        log.error(f"Action execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Status and monitoring routes
@status_router.get("/overview")
async def get_system_overview():
    """Get system overview and status."""
    try:
        droidbot = get_droidbot()
        
        # Get device status
        device_status = await droidbot.get_device_status()
        
        # Get session info
        session_info = droidbot.get_session_info()
        
        # Get task history
        task_history = await droidbot.get_task_history()
        
        return {
            "device": device_status,
            "session": session_info,
            "tasks": {
                "total": len(task_history),
                "recent": task_history[-5:] if task_history else []
            },
            "system": {
                "status": "running",
                "version": "1.0.0"
            }
        }
        
    except Exception as e:
        log.error(f"System overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@status_router.get("/session")
async def get_session_info():
    """Get current session information."""
    try:
        droidbot = get_droidbot()
        session_info = droidbot.get_session_info()
        return session_info
        
    except Exception as e:
        log.error(f"Session info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@status_router.get("/resources")
async def get_resource_usage():
    """Get current resource usage."""
    try:
        droidbot = get_droidbot()
        
        if not droidbot.device_manager.is_connected():
            raise HTTPException(status_code=400, detail="Device not connected")
            
        resource_usage = await droidbot.device_manager.get_resource_usage()
        return resource_usage
        
    except Exception as e:
        log.error(f"Resource usage error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 