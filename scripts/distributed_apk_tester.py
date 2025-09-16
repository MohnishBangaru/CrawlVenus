#!/usr/bin/env python3
"""
Distributed Universal APK Tester for RunPod + Local Emulator
============================================================

This script runs on RunPod and communicates with your local laptop's ADB server
to test Android APKs. The AI processing happens on RunPod while device interaction
happens on your local emulator.

Usage:
    python scripts/distributed_apk_tester.py --apk /path/to/app.apk --local-server http://YOUR_LAPTOP_IP:8000
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.distributed_device_manager import DistributedDeviceManager, test_distributed_connection
from src.core.distributed_config import distributed_config
from src.vision.engine import VisionEngine
from src.ai.phi_ground import PhiGroundActionGenerator
from src.ai.openai_client import OpenAIClient
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedActionExecutor:
    """Action executor that works with DistributedDeviceManager."""
    
    def __init__(self, device_manager: DistributedDeviceManager):
        """Initialize the distributed action executor."""
        self.device_manager = device_manager
        self.action_history = []
        self.max_retries = 3
    
    async def execute_action(self, action: dict) -> bool:
        """Execute a single action using the distributed device manager."""
        action_type = action.get("type")
        action_id = f"{action_type}_{int(time.time() * 1000)}"
        
        logger.info(f"Executing action {action_id}: {action_type}")
        
        for attempt in range(self.max_retries + 1):
            try:
                success = self._perform_action(action)
                if success:
                    logger.info(f"Action {action_id} completed successfully")
                    return True
                    
                if attempt < self.max_retries:
                    logger.warning(f"Action failed, retrying ({attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Action error, retrying ({attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.error(f"Action failed after {self.max_retries} retries: {e}")
                    
        return False
    
    def _perform_action(self, action: dict) -> bool:
        """Perform the actual action using the device manager's specific methods."""
        action_type = action.get("type")
        
        if action_type == "tap":
            return self.device_manager.tap(action["x"], action["y"])
        elif action_type == "swipe":
            return self.device_manager.swipe(
                action["start_x"], action["start_y"],
                action["end_x"], action["end_y"],
                action.get("duration", 500)
            )
        elif action_type == "keyevent":
            return self.device_manager.send_keyevent(action["key_code"])
        elif action_type == "input_text":
            # For text input, we need to use shell command
            text = action["text"]
            command = f"input text '{text}'"
            result = self.device_manager.execute_shell_command(command)
            return result is not None
        elif action_type == "wait":
            # For wait actions, just sleep
            duration = action.get("duration", 1.0)
            time.sleep(duration)
            return True
        else:
            logger.error(f"Unknown action type: {action_type}")
            return False


class DistributedAPKTester:
    """Distributed APK tester for RunPod + Local Emulator setup."""
    
    def __init__(self, apk_path: str, local_server_url: str, output_dir: str = "test_reports"):
        """Initialize distributed APK tester."""
        self.apk_path = apk_path
        self.local_server_url = local_server_url
        
        # Use specified local directory for saving results (on laptop, not RunPod)
        if output_dir.startswith("http"):
            # If using tunnel URL, save to specified local directory
            self.output_dir = Path("/Users/mohnishbangaru/Drizz/local_test_reports")
        else:
            # Use the specified output directory
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.device_manager = None
        self.vision_engine = None
        self.phi_ground = None
        self.openai_client = None
        self.action_executor = None
        
        # Test results
        self.test_results = {
            "start_time": None,
            "end_time": None,
            "actions_performed": 0,
            "screenshots_taken": 0,
            "errors": [],
            "success": False
        }
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Distributed APK Tester...")
        
        # Test connection to local ADB server
        if not await test_distributed_connection(self.local_server_url):
            raise ConnectionError(f"Cannot connect to local ADB server at {self.local_server_url}")
        
        # Initialize device manager
        self.device_manager = DistributedDeviceManager(self.local_server_url)
        
        # Initialize vision engine
        self.vision_engine = VisionEngine()
        
        # Initialize AI components
        try:
            self.phi_ground = PhiGroundActionGenerator()
            await self.phi_ground.initialize()
            logger.info("Phi Ground initialized successfully")
        except Exception as e:
            logger.warning(f"Phi Ground initialization failed: {e}")
            self.phi_ground = None
        
        try:
            self.openai_client = OpenAIClient()
            await self.openai_client.initialize()
            logger.info("OpenAI Client initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI Client initialization failed: {e}")
            self.openai_client = None
        
        # Initialize action executor
        self.action_executor = DistributedActionExecutor(self.device_manager)
        
        logger.info("Distributed APK Tester initialized successfully")
    
    async def run_test(self, num_actions: int = 10):
        """Run the APK test."""
        logger.info(f"Starting APK test with {num_actions} actions")
        self.test_results["start_time"] = time.time()
        
        try:
            # Wait for device
            if not self.device_manager.wait_for_device():
                raise RuntimeError("Device not available")
            
            # Optimize device settings
            self.device_manager.optimize_device_settings()
            
            # Install APK
            if not self.device_manager.install_apk(self.apk_path):
                raise RuntimeError("Failed to install APK")
            
            # Extract package name from APK path
            package_name = self._extract_package_name(self.apk_path)
            
            # Launch app
            if not self.device_manager.launch_app(package_name):
                raise RuntimeError("Failed to launch app")
            
            # Wait for app to load
            await asyncio.sleep(3)
            
            # Perform actions
            for i in range(num_actions):
                logger.info(f"Performing action {i+1}/{num_actions}")
                
                try:
                    # Take screenshot
                    screenshot = self.device_manager.take_screenshot()
                    if screenshot:
                        screenshot_path = await self._save_screenshot_locally(screenshot, f"action_{i+1}_screenshot.png")
                        if screenshot_path:
                            self.test_results["screenshots_taken"] += 1
                            
                            # Analyze screenshot with vision engine
                            elements = self.vision_engine.analyze(screenshot_path)
                            
                            # Generate action using AI
                            action = await self._generate_action(screenshot_path, elements)
                        else:
                            logger.error(f"Failed to save action {i+1} screenshot")
                            continue
                        
                        if action:
                            # Execute action
                            success = await self.action_executor.execute_action(action)
                            if success:
                                self.test_results["actions_performed"] += 1
                            
                            # Wait for action to complete
                            await asyncio.sleep(1)
                        else:
                            logger.warning("No action generated, skipping")
                    else:
                        logger.error("Failed to take screenshot")
                        
                except Exception as e:
                    logger.error(f"Error during action {i+1}: {e}")
                    self.test_results["errors"].append(f"Action {i+1}: {str(e)}")
            
            # Take final screenshot
            final_screenshot = self.device_manager.take_screenshot()
            if final_screenshot:
                screenshot_path = await self._save_screenshot_locally(final_screenshot, "final_screenshot.png")
                if screenshot_path:
                    self.test_results["screenshots_taken"] += 1
                else:
                    logger.error("Failed to save final screenshot")
            else:
                logger.error("Failed to take final screenshot")
            
            # Uninstall app
            self.device_manager.uninstall_app(package_name)
            
            self.test_results["success"] = True
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            self.test_results["errors"].append(f"Test failure: {str(e)}")
        
        finally:
            self.test_results["end_time"] = time.time()
            await self._save_results()
    
    async def _generate_action(self, screenshot_path: str, elements: list) -> Optional[dict]:
        """Generate action using AI."""
        try:
            if self.phi_ground:
                # Use Phi Ground for action generation
                action = await self.phi_ground.generate_touch_action(
                    screenshot_path, 
                    "Automate app interaction", 
                    [],  # Empty action history for now
                    elements
                )
                return action
            elif self.openai_client:
                # Fallback to OpenAI
                action = await self.openai_client.generate_action(screenshot_path, elements)
                return action
            else:
                # Random action as last resort
                return self._generate_random_action()
        except Exception as e:
            logger.error(f"Failed to generate action: {e}")
            return self._generate_random_action()
    
    def _generate_random_action(self) -> dict:
        """Generate a random action with improved coordinate selection."""
        import random
        
        action_types = ["tap", "swipe", "keyevent"]
        action_type = random.choice(action_types)
        
        if action_type == "tap":
            # Use more intelligent coordinate ranges
            # Avoid status bar (top 100px) and navigation bar (bottom 200px)
            return {
                "type": "tap",
                "x": random.randint(150, 850),  # Avoid screen edges
                "y": random.randint(200, 1000),  # Avoid status and navigation bars
                "reasoning": "Smart random fallback - avoiding system UI areas",
                "source": "improved_random"
            }
        elif action_type == "swipe":
            return {
                "type": "swipe",
                "start_x": random.randint(100, 400),
                "start_y": random.randint(200, 600),
                "end_x": random.randint(500, 800),
                "end_y": random.randint(700, 1200),
                "duration": random.randint(300, 800)
            }
        else:  # keyevent
            return {
                "type": "keyevent",
                "key_code": random.choice([4, 24, 25])  # Back, Volume Up, Volume Down
            }
    
    def _extract_package_name(self, apk_path: str) -> str:
        """Extract package name from APK filename."""
        # For now, use a simple heuristic based on filename
        # In a real implementation, you'd use aapt or similar to extract package info
        filename = Path(apk_path).name
        
        # Try to extract package name from filename like "com.Dominos_12.1.16-299_minAPI23..."
        if "com." in filename:
            package_part = filename.split("_")[0]
            if package_part.startswith("com."):
                return package_part
        
        # Fallback: try to extract from common patterns
        if "Dominos" in filename:
            return "com.Dominos"
        
        # Default fallback
        return "com.example.app"
    
    async def _save_screenshot_locally(self, screenshot_data: bytes, filename: str) -> Optional[str]:
        """Save screenshot to local directory."""
        try:
            screenshot_path = self.output_dir / filename
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot_data)
            return str(screenshot_path)
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return None
    
    async def _save_results(self):
        """Save test results to local directory."""
        try:
            results_path = self.output_dir / "test_results.json"
            
            # Calculate duration
            duration = self.test_results["end_time"] - self.test_results["start_time"]
            
            # Create summary
            summary = {
                "test_info": {
                    "apk_path": self.apk_path,
                    "local_server": self.local_server_url,
                    "duration_seconds": duration,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": self.test_results,
                "success_rate": self.test_results["actions_performed"] / max(1, len(self.test_results["errors"]) + self.test_results["actions_performed"])
            }
            
            import json
            with open(results_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Results saved to: {self.output_dir}")
            
            # Print summary
            logger.info(f"Test completed in {duration:.2f} seconds")
            logger.info(f"Actions performed: {self.test_results['actions_performed']}")
            logger.info(f"Screenshots taken: {self.test_results['screenshots_taken']}")
            logger.info(f"Errors: {len(self.test_results['errors'])}")
            logger.info(f"Success: {self.test_results['success']}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Distributed APK Tester")
    parser.add_argument("--apk", required=True, help="Path to APK file")
    parser.add_argument("--local-server", required=True, help="Local ADB server URL")
    parser.add_argument("--actions", type=int, default=10, help="Number of actions to perform")
    parser.add_argument("--output-dir", default="test_reports", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create tester
    tester = DistributedAPKTester(args.apk, args.local_server, args.output_dir)
    
    try:
        # Initialize
        await tester.initialize()
        
        # Run test
        await tester.run_test(args.actions)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
