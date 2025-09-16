#!/usr/bin/env python3
"""Universal APK Tester with DroidBot-GPT + VisionAI

This script provides a universal testing solution for any APK:
1. Accepts APK path and number of actions as command line arguments
2. Installs and launches the APK
3. Performs specified number of actions with screenshots
4. Runs VisionAI analysis
5. Generates comprehensive final report
"""

import json
import os
import random
import re
import subprocess
import sys
import time
import hashlib  # For screenshot hashing to detect state changes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from scripts.final_report_generator import AppState, FinalReportGenerator, LLMEvent


class UniversalAPKTester:
    """Universal APK testing with Explorer-GPT + VisionAI integration."""
    
    def __init__(self, apk_path: str, num_actions: int = 10, output_dir: str = "universal_reports", 
                 screenshot_frequency: str = "adaptive", max_screenshots_per_action: int = 3,
                 generate_reports: bool = True):
        """Initialize Universal APK Tester.
        
        Parameters
        ----------
        apk_path : str
            Path to the APK file
        num_actions : int
            Number of actions to perform
        output_dir : str
            Output directory for reports
        screenshot_frequency : str
            Screenshot frequency mode: "minimal", "adaptive", "high"
        max_screenshots_per_action : int
            Maximum screenshots per action (for adaptive mode)
        generate_reports : bool
            Whether to generate reports

        """
        self.apk_path = Path(apk_path)
        self.num_actions = num_actions
        self.output_dir = Path(output_dir)
        self.screenshot_frequency = screenshot_frequency
        self.max_screenshots_per_action = max_screenshots_per_action
        self.generate_reports = generate_reports
        
        # Extract app information
        self.package_name = self._extract_package_name()
        self.app_name = self._extract_app_name()
        
        # Cache screen size for coordinate scaling
        self.screen_width, self.screen_height = self._get_screen_size()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir = self.output_dir / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.action_count = 0
        self.current_state_id = "state_0"
        self.app_states: Dict[str, AppState] = {}
        self.llm_events: List[LLMEvent] = []
        self.testing_completed = False
        # Track hash of last state screenshot to detect state changes
        self._last_screenshot_hash: str | None = None
        # Track last screenshot path for Phi Ground
        self._last_screenshot_path: str | None = None
        # Count only actions that cause a UI change
        self.changed_action_count = 0
        
        # Initialize VisionAI engine
        try:
            from src.vision.engine import VisionEngine
            self.vision_engine = VisionEngine()
            logger.info("✓ VisionAI engine initialized")
        except Exception as e:
            logger.warning(f"VisionAI engine initialization failed: {e}")
            self.vision_engine = None
        
        logger.info("Initialized Universal APK Tester")
        logger.info(f"APK: {self.apk_path}")
        logger.info(f"App: {self.app_name}")
        logger.info(f"Package: {self.package_name}")
        logger.info(f"Actions: {self.num_actions}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Screenshot Frequency: {self.screenshot_frequency}")
        logger.info(f"Max Screenshots per Action: {self.max_screenshots_per_action}")
        logger.info(f"Generate reports: {self.generate_reports}")
    
    def _extract_package_name(self) -> str:
        """Extract package name from APK filename."""
        filename = self.apk_path.name
        
        # Common patterns
        if filename.startswith("com.Dominos"):
            return "com.Dominos"
        elif filename.startswith("com.yum.pizzahut"):
            return "com.yum.pizzahut"
        elif filename.startswith("com."):
            # Extract first part before underscore
            parts = filename.split('_')
            if len(parts) > 0:
                return parts[0]
        
        # Fallback: use `aapt dump badging` if available to inspect APK metadata
        try:
            import shlex, subprocess, re
            aapt_cmd = "aapt"
            # If ANDROID_HOME is set, try build-tools path first
            android_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
            if android_home:
                # pick latest build-tools dir
                build_tools = Path(android_home) / "build-tools"
                if build_tools.exists():
                    versions = sorted(build_tools.iterdir(), key=lambda p: p.name, reverse=True)
                    for v in versions:
                        candidate = v / "aapt"
                        if candidate.exists():
                            aapt_cmd = str(candidate)
                            break

            result = subprocess.run(
                shlex.split(f"{aapt_cmd} dump badging '{self.apk_path}'"),
                capture_output=True,
                text=True,
                check=False,
            )
            m = re.search(r"package: name='([^']+)'", result.stdout)
            if m:
                return m.group(1)
        except Exception:
            pass
        
        return "unknown.package"
    
    def _extract_app_name(self) -> str:
        """Extract app name from APK filename."""
        filename = self.apk_path.name
        
        if "Dominos" in filename:
            return "Domino's Pizza"
        elif "pizzahut" in filename:
            return "Pizza Hut"
        elif "com." in filename:
            # Extract app name from package
            package = self._extract_package_name()
            if package.startswith("com."):
                parts = package.split('.')
                if len(parts) > 1:
                    return parts[-1].title()
        
        return "Unknown App"
    
    def check_environment(self) -> bool:
        """Check if all required components are available.
        
        Returns
        -------
        bool
            True if environment is ready, False otherwise

        """
        logger.info("Checking environment...")
        
        # Check ADB
        try:
            result = subprocess.run(["adb", "version"], capture_output=True, text=True, check=True)
            logger.info("✓ ADB available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("✗ ADB not found")
            return False
        
        # Check device connection
        try:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]
            connected_devices = [line for line in lines if line.strip() and 'device' in line]
            
            if connected_devices:
                logger.info(f"✓ Found {len(connected_devices)} connected device(s)")
                for device in connected_devices:
                    logger.info(f"  - {device}")
            else:
                logger.error("✗ No devices connected")
                return False
        except subprocess.CalledProcessError:
            logger.error("✗ Failed to check devices")
            return False
        
        # Check APK file
        if not self.apk_path.exists():
            logger.error(f"✗ APK not found: {self.apk_path}")
            return False
        logger.info("✓ APK file exists")
        
        return True
    
    def install_apk(self) -> bool:
        """Install the APK on the device.
        
        Returns
        -------
        bool
            True if installation successful

        """
        logger.info(f"Installing {self.app_name} APK...")
        
        try:
            # Uninstall if exists
            subprocess.run(
                ["adb", "uninstall", self.package_name],
                capture_output=True,
                check=False
            )
            
            # Install using ADB
            result = subprocess.run(
                ["adb", "install", "-r", str(self.apk_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            if "Success" in result.stdout:
                logger.info("✓ APK installed successfully")
                return True
            else:
                logger.error(f"Installation failed: {result.stdout}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Installation error: {e}")
            return False
    
    def launch_app(self) -> bool:
        """Launch the app with proper popup handling and stabilization.
        
        Returns
        -------
        bool
            True if launch successful

        """
        logger.info(f"Launching {self.app_name}...")
        
        try:
            # Step 1: Launch the app
            logger.info("Step 1: Launching app...")
            result = subprocess.run(
                ["adb", "shell", "monkey", "-p", self.package_name, "-c", "android.intent.category.LAUNCHER", "1"],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("✓ App launched successfully")
            
            # Step 2: Wait for initial load and handle popups
            logger.info("Step 2: Waiting for app to stabilize and handling popups...")
            time.sleep(3)  # Initial wait for app to start loading
            
            # Step 3: Handle common popups during launch
            logger.info("Step 3: Handling launch popups...")
            self._handle_launch_popups()
            
            # Step 4: Wait for app to fully stabilize
            logger.info("Step 4: Waiting for app to fully stabilize...")
            time.sleep(3)  # Additional wait for app to fully load
            
            # Step 5: Verify app is in foreground (but don't fail if it's not yet)
            logger.info("Step 5: Verifying app launch...")
            if self.check_app_foreground():
                logger.info("✓ App is in foreground after launch")
            else:
                logger.info("App launched but not yet in foreground, this is normal during launch sequence")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Launch error: {e}")
            return False
    
    def _handle_launch_popups(self) -> None:
        """Handle common popups that appear during app launch."""
        logger.info("Handling launch popups...")
        
        # Common popup coordinates and actions
        popup_actions = [
            # Google Play Services popups
            {"action": "tap", "coords": (672, 1200)},  # Common "OK" button location
            {"action": "tap", "coords": (672, 1400)},  # Common "Allow" button location
            {"action": "tap", "coords": (672, 1600)},  # Common "Continue" button location
            
            # App-specific popups
            {"action": "tap", "coords": (672, 1000)},  # Common "Get Started" location
            {"action": "tap", "coords": (672, 1100)},  # Common "Skip" button location
            {"action": "tap", "coords": (672, 1300)},  # Common "Next" button location
            
            # "Turn On" and "Dismiss" buttons
            {"action": "tap", "coords": (672, 1500)},  # "Turn On" button
            {"action": "tap", "coords": (672, 1700)},  # "Dismiss" button
            
            # Back button to dismiss any remaining dialogs
            {"action": "keyevent", "key": "4"},  # Back button
        ]
        
        for i, popup_action in enumerate(popup_actions):
            try:
                if popup_action["action"] == "tap":
                    self._tap(*popup_action["coords"])
                elif popup_action["action"] == "keyevent":
                    self._tap(*popup_action["key"])
                
                time.sleep(0.5)  # Brief pause between popup actions
                
            except Exception as e:
                logger.debug(f"Popup action {i+1} failed: {e}")
        
        logger.info("✓ Launch popup handling completed")
    
    def keep_app_foreground(self) -> bool:
        """Keep the app in foreground during testing with improved reliability.
        
        Returns
        -------
        bool
            True if successful

        """
        logger.info(f"Keeping {self.app_name} in foreground...")
        
        try:
            # Method 1: Use monkey to launch app (most reliable)
            logger.info("Method 1: Using monkey to launch app...")
            result1 = subprocess.run(
                ["adb", "shell", "monkey", "-p", self.package_name, "-c", "android.intent.category.LAUNCHER", "1"],
                capture_output=True,
                text=True,
                check=False
            )
            time.sleep(1)
            
            if self.check_app_foreground():
                logger.info("✓ App brought to foreground with monkey")
                return True
            
            # Method 2: Use am start with launcher intent
            logger.info("Method 2: Using am start with launcher intent...")
            result2 = subprocess.run(
                ["adb", "shell", "am", "start", "-W", "-a", "android.intent.action.MAIN", 
                 "-c", "android.intent.category.LAUNCHER", "-n", f"{self.package_name}/.MainActivity"],
                capture_output=True,
                text=True,
                check=False
            )
            time.sleep(1)
            
            if self.check_app_foreground():
                logger.info("✓ App brought to foreground with am start")
                return True
            
            # Method 3: Try to find the correct activity name
            logger.info("Method 3: Finding correct activity name...")
            result3 = subprocess.run(
                ["adb", "shell", "cmd", "package", "resolve-activity", "--brief", self.package_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result3.stdout and "activity" in result3.stdout:
                # Extract activity name from output
                lines = result3.stdout.strip().split('\n')
                for line in lines:
                    if self.package_name in line and "activity" in line:
                        activity_name = line.split()[-1]
                        logger.info(f"Found activity: {activity_name}")
                        
                        result4 = subprocess.run(
                            ["adb", "shell", "am", "start", "-W", "-n", activity_name],
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        time.sleep(1)
                        
                        if self.check_app_foreground():
                            logger.info("✓ App brought to foreground with correct activity")
                            return True
                        break
            
            # Method 4: Force bring to front using input
            logger.info("Method 4: Using input to bring app to front...")
            subprocess.run(
                ["adb", "shell", "input", "keyevent", "KEYCODE_APP_SWITCH"],
                capture_output=True,
                check=False
            )
            time.sleep(0.5)
            
            # Try to tap on the app in recent apps
            self._tap(540, 1000)  # Coordinate tap (scaled)
            time.sleep(1)
            
            if self.check_app_foreground():
                logger.info("✓ App brought to foreground with input method")
                return True
            
            # Method 5: Last resort - restart the app completely
            logger.info("Method 5: Restarting app completely...")
            subprocess.run(
                ["adb", "shell", "am", "force-stop", self.package_name],
                capture_output=True,
                check=False
            )
            time.sleep(1)
            
            subprocess.run(
                ["adb", "shell", "monkey", "-p", self.package_name, "-c", "android.intent.category.LAUNCHER", "1"],
                capture_output=True,
                check=False
            )
            time.sleep(2)
            
            if self.check_app_foreground():
                logger.info("✓ App restarted and brought to foreground")
                return True
            
            logger.warning("All foreground methods failed")
            return False
            
        except Exception as e:
            logger.warning(f"Foreground operation failed: {e}")
            return False
    
    def relaunch_app_if_needed(self) -> bool:
        """Relaunch the app if it's not in foreground.
        
        Returns
        -------
        bool
            True if app was relaunched or is already in foreground

        """
        logger.info("Checking if app needs to be relaunched...")
        
        # Check if app is in foreground
        if self.check_app_foreground():
            logger.info("✓ App is already in foreground")
            return True
        
        # Check if app is running at all
        try:
            result = subprocess.run(
                ["adb", "shell", "ps", "|", "grep", self.package_name],
                capture_output=True,
                text=True,
                shell=True,
                check=False
            )
            
            if self.package_name in result.stdout:
                logger.info("App is running but not in foreground, attempting to bring to front...")
                return self.keep_app_foreground()
            else:
                logger.info("App is not running, relaunching...")
                return self.launch_app()
                
        except Exception as e:
            logger.warning(f"Error checking app status: {e}")
            logger.info("Attempting to relaunch app...")
            return self.launch_app()
    
    def ensure_app_active(self, max_retries: int = 3) -> bool:
        """Ensure the app is active and in foreground with automatic relaunch.
        
        Parameters
        ----------
        max_retries : int
            Maximum number of relaunch attempts
            
        Returns
        -------
        bool
            True if app is active and in foreground

        """
        for attempt in range(max_retries):
            logger.info(f"Ensuring app is active (attempt {attempt + 1}/{max_retries})...")
            
            # Try to bring app to foreground or relaunch if needed
            if self.relaunch_app_if_needed():
                # Double-check that app is actually in foreground
                if self.check_app_foreground():
                    logger.info("✓ App is active and in foreground")
                    return True
                else:
                    logger.warning(f"App relaunch succeeded but not in foreground (attempt {attempt + 1})")
            else:
                logger.warning(f"App relaunch failed (attempt {attempt + 1})")
            
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2)
        
        logger.error("Failed to ensure app is active after all attempts")
        return False
    
    def enhanced_foreground_recovery(self) -> bool:
        """Enhanced foreground recovery with single strategy - direct app launch."""
        logger.info("Running enhanced foreground recovery with single strategy...")
        
        # Check current foreground app for debugging
        current_app = self.get_current_foreground_app()
        logger.info(f"Current foreground app: {current_app}")
        logger.info(f"Target app: {self.package_name}")
        
        # First, try to handle popups specifically
        if self._handle_popups():
            time.sleep(2)
            if self.check_app_foreground():
                logger.info("✓ Popup handling succeeded")
                return True
            else:
                logger.warning("Popup handling completed but app still not in foreground")
        
        # Single recovery strategy: Direct app launch
        logger.info("Trying single recovery strategy: Direct app launch...")
        try:
            # Use the direct app launch method
            success = self._try_direct_app_launch_recovery()
            
            if success:
                time.sleep(1)  # Reduced delay for faster recovery
                if self.check_app_foreground():
                    logger.info("✓ Single recovery strategy succeeded")
                    return True
                else:
                    current_app = self.get_current_foreground_app()
                    logger.warning(f"Single recovery strategy completed but app still not in foreground. Current app: {current_app}")
            else:
                logger.warning("Single recovery strategy failed")
                
        except Exception as e:
            logger.warning(f"Single recovery strategy failed with error: {e}")
        
        logger.error("❌ Single recovery strategy failed")
        return False
    
    def get_current_foreground_app(self) -> str:
        """Get the current foreground app package name."""
        try:
            # Method 1: Check mResumedActivity
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "activity", "activities"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'mResumedActivity' in line:
                        # Extract package name from the line
                        import re
                        match = re.search(r'(\S+)/\S+', line)
                        if match:
                            return match.group(1)
            
            # Method 2: Check mCurrentFocus
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "window", "windows"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'mCurrentFocus' in line:
                        import re
                        match = re.search(r'(\S+)/\S+', line)
                        if match:
                            return match.group(1)
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error getting current foreground app: {e}")
            return "unknown"
    
    def _handle_popups(self) -> bool:
        """Handle common popup scenarios for apps with intelligent classification."""
        logger.info("Handling popup scenarios with intelligent classification...")
        
        # External popup indicators (system dialogs, permissions) - should be dismissed
        external_popup_indicators = {
            "system": ["google play", "android", "system", "settings", "permission"],
            "permission": ["allow", "deny", "permission", "access", "grant"],
            "notification": ["notification", "alert", "warning", "error", "update"],
            "external_app": ["chrome", "browser", "gmail", "maps", "play store"]
        }
        
        # Internal popup indicators (menu cards, navigation) - should be handled through decision-making
        internal_popup_indicators = {
            "menu": ["menu", "options", "more", "settings", "profile", "account"],
            "navigation": ["back", "forward", "home", "tab", "page", "section"],
            "selection": ["select", "choose", "pick", "option", "item", "card"],
            "app_specific": ["order", "cart", "checkout", "payment", "delivery", "tracking"]
        }
        
        # Menu card specific patterns
        menu_card_patterns = [
            r"menu.*card",
            r"card.*menu", 
            r"option.*card",
            r"item.*card",
            r"selection.*card",
            r"choice.*card"
        ]
        
        # Navigation element patterns
        navigation_patterns = [
            r"back.*button",
            r"forward.*button", 
            r"home.*button",
            r"tab.*bar",
            r"navigation.*menu",
            r"breadcrumb"
        ]
        
        # System popup patterns
        system_popup_patterns = [
            r"google.*play",
            r"android.*system",
            r"permission.*dialog",
            r"system.*alert",
            r"external.*app"
        ]
        
        def classify_popup_element(text: str, element_type: str) -> str:
            """Classify popup element as internal or external."""
            text_lower = text.lower()
            
            # Check for menu card patterns
            for pattern in menu_card_patterns:
                if re.search(pattern, text_lower):
                    return "internal_menu_card"
            
            # Check for navigation patterns
            for pattern in navigation_patterns:
                if re.search(pattern, text_lower):
                    return "internal_navigation"
            
            # Check for system popup patterns
            for pattern in system_popup_patterns:
                if re.search(pattern, text_lower):
                    return "external_system"
            
            # Check internal indicators
            for category, keywords in internal_popup_indicators.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return "internal_app"
            
            # Check external indicators
            for category, keywords in external_popup_indicators.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return "external_system"
            
            # Check element type
            if element_type in ['card', 'menu_item', 'option']:
                return "internal_menu_card"
            elif element_type in ['back_button', 'navigation', 'tab']:
                return "internal_navigation"
            elif element_type in ['system_dialog', 'permission_dialog', 'external_dialog']:
                return "external_system"
            
            return "unknown"
        
        def has_clickable_elements(text: str) -> bool:
            """Check if popup has clickable elements other than close/dismiss."""
            text_lower = text.lower()
            
            # Clickable action keywords (not close/dismiss)
            action_keywords = [
                "continue", "proceed", "next", "submit", "confirm", "accept", "allow",
                "select", "choose", "pick", "option", "item", "menu", "settings",
                "order", "cart", "checkout", "payment", "delivery", "tracking"
            ]
            
            # Close/dismiss keywords (should be ignored for this check)
            close_keywords = ["close", "dismiss", "cancel", "back", "exit", "ok"]
            
            # Check for action keywords (indicating clickable elements)
            for keyword in action_keywords:
                if keyword in text_lower:
                    return True
            
            # Check for close/dismiss keywords (no other clickable elements)
            has_close_only = any(keyword in text_lower for keyword in close_keywords)
            
            return not has_close_only  # If only close/dismiss, no other clickable elements
        
        # Google Play Services popup handling (external - dismiss)
        # More targeted approach to avoid affecting app navigation
        external_popup_actions = [
            {"action": "tap", "coords": (500, 1200), "desc": "Google Play OK button", "type": "external_system"},
            {"action": "tap", "coords": (500, 1100), "desc": "Google Play Accept button", "type": "external_system"},
            {"action": "tap", "coords": (500, 1300), "desc": "Google Play Continue button", "type": "external_system"},
        ]
        
        # Permission dialog handling (external - dismiss)
        permission_popup_actions = [
            {"action": "tap", "coords": (500, 1200), "desc": "Permission Turn On button", "type": "external_system"},
            {"action": "tap", "coords": (500, 1100), "desc": "Permission Accept button", "type": "external_system"},
            {"action": "tap", "coords": (500, 1300), "desc": "Permission Allow button", "type": "external_system"},
            {"action": "tap", "coords": (400, 1200), "desc": "Permission left button", "type": "external_system"},
            {"action": "tap", "coords": (600, 1200), "desc": "Permission right button", "type": "external_system"},
        ]
        
        # Dismiss button handling (external - dismiss) - more conservative
        dismiss_actions = [
            {"action": "tap", "coords": (500, 1200), "desc": "Dismiss button (bottom center)", "type": "external_system"},
            {"action": "tap", "coords": (500, 1100), "desc": "Dismiss button (above bottom)", "type": "external_system"},
        ]
        
        # Internal popup actions (menu cards, navigation) - should be handled through decision-making
        internal_popup_actions = [
            {"action": "tap", "coords": (500, 800), "desc": "Internal menu Continue button", "type": "internal_menu_card"},
            {"action": "tap", "coords": (500, 900), "desc": "Internal menu Accept button", "type": "internal_menu_card"},
            {"action": "tap", "coords": (400, 800), "desc": "Internal menu left button", "type": "internal_menu_card"},
            {"action": "tap", "coords": (600, 800), "desc": "Internal menu right button", "type": "internal_menu_card"},
            {"action": "tap", "coords": (500, 1000), "desc": "Internal menu bottom button", "type": "internal_menu_card"},
        ]
        
        # Execute external popup actions (dismiss) with foreground checking
        logger.info("Handling external popups (system dialogs, permissions) - dismissing...")
        
        # Track if we successfully dismissed any external popups
        external_popup_dismissed = False
        
        for action in external_popup_actions + permission_popup_actions + dismiss_actions:
            logger.info(f"Attempting to dismiss external popup: {action['desc']}")
            self._tap(*action['coords'])
            time.sleep(0.3)
            
            # Check if app is still in foreground after tap
            if self.check_app_foreground():
                external_popup_dismissed = True
                logger.info(f"Successfully dismissed external popup: {action['desc']}")
            else:
                logger.warning(f"App left foreground after dismissing: {action['desc']}")
                break
        
        # Note: Internal popups (menu cards) are NOT dismissed here
        # They should be handled through the decision-making process
        logger.info("Internal popups (menu cards, navigation) will be handled through decision-making process")
        
        # Look for Close/Dismiss buttons on external popups ONLY if we're certain it's external
        # This is more conservative to avoid dismissing internal menu cards
        if not external_popup_dismissed:
            logger.info("Checking if we should look for Close/Dismiss buttons...")
            
            # Only proceed if we're confident this is an external popup
            # Check if we have any clear external popup indicators
            current_foreground = self.get_current_foreground_app()
            has_external_indicators = any(indicator in current_foreground.lower() for indicator in 
                                        ["google", "android", "system", "settings", "permission", "chrome", "browser"])
            
            if has_external_indicators:
                logger.info("External popup detected, looking for Close/Dismiss buttons...")
                
                # Common Close/Dismiss button locations for external popups
                close_dismiss_buttons = [
                    {"coords": (900, 200), "desc": "Close button (top right)"},
                    {"coords": (800, 200), "desc": "Close button (top right area)"},
                    {"coords": (500, 200), "desc": "Close button (top center)"},
                    {"coords": (500, 1200), "desc": "Dismiss button (bottom center)"},
                    {"coords": (500, 1100), "desc": "Dismiss button (above bottom)"},
                    {"coords": (300, 1200), "desc": "Cancel button (bottom left)"},
                    {"coords": (700, 1200), "desc": "Cancel button (bottom right)"},
                ]
                
                # Try each Close/Dismiss button location
                for button in close_dismiss_buttons:
                    logger.info(f"Trying {button['desc']} on external popup")
                    self._tap(*button['coords'])
                    time.sleep(0.3)
                    
                    # Check if app is still in foreground after tap
                    if self.check_app_foreground():
                        logger.info(f"Successfully used {button['desc']} to dismiss external popup")
                        break
                    else:
                        logger.warning(f"App left foreground after {button['desc']}, stopping")
                        break
            else:
                logger.info("No clear external popup indicators found, skipping Close/Dismiss button attempts")
                logger.info("Internal popups (menu cards, navigation) should be handled through decision-making process")
        
        # Enhanced location permission prompt handling using OCR
        try:
            # Capture screenshot for OCR analysis
            screenshot_path = self.capture_screenshot("permission_detection")
            if screenshot_path:
                # Use VisionAI to analyze the screenshot
                elements = self.vision_engine.analyze(screenshot_path)
                
                # Look for permission-related buttons
                permission_keywords = [
                    "allow while using the app", "allow only while using the app", "allow all the time",
                    "while using the app", "only while using the app", "all the time",
                    "allow", "ok", "continue", "accept", "grant", "permit", "enable"
                ]
                
                for element in elements:
                    element_text = element.text.lower() if hasattr(element, 'text') else ""
                    
                    # Check if this looks like a permission button
                    if any(keyword in element_text for keyword in permission_keywords):
                        # Get coordinates and tap
                        bbox = element.bbox.as_tuple() if hasattr(element, 'bbox') else []
                        if len(bbox) >= 4:
                            x = (bbox[0] + bbox[2]) // 2
                            y = (bbox[1] + bbox[3]) // 2
                            logger.info(f"Tapping permission button via OCR: {element_text}")
                            self._tap(x, y)
                            time.sleep(0.5)  # Slightly longer delay for permission dialogs
                            
                            # Check if we successfully dismissed the permission dialog
                            if self.check_app_foreground():
                                logger.info(f"✓ Permission dialog dismissed with OCR: {element_text}")
                                return True
                
                logger.warning("No permission buttons found via OCR")
            else:
                logger.warning("Failed to capture screenshot for OCR permission detection")
                
        except Exception as e:
            logger.warning(f"Error in OCR permission handling: {e}")
            # Fallback to original coordinates
            fallback_actions = [
                {"coords": (500, 1200), "desc": "Allow While Using the App button"},
                {"coords": (500, 1100), "desc": "Allow Only While Using the App button"},
                {"coords": (500, 1300), "desc": "Allow All the Time button"},
                {"coords": (500, 1000), "desc": "Allow button"},
                {"coords": (500, 900), "desc": "OK button"},
                {"coords": (500, 1000), "desc": "Continue button"},
            ]
            for action in fallback_actions:
                logger.info(f"Trying fallback: {action['desc']}")
                self._tap(*action['coords'])
                time.sleep(0.3)
        
        # ----------------------
        # Vision-based "×" close button detection (top-right glyph)
        # ----------------------
        try:
            screenshot_path = self.capture_screenshot("close_button_detection")
            if screenshot_path:
                elements = self.vision_engine.analyze(screenshot_path)

                for element in elements:
                    txt = (element.text or "").strip().lower() if hasattr(element, "text") else ""
                    if txt in {"x", "×", "close"} and hasattr(element, "bbox"):
                        bx = element.bbox.as_tuple()
                        cx = (bx[0] + bx[2]) // 2
                        cy = (bx[1] + bx[3]) // 2
                        # consider only glyphs near the top-right quadrant
                        if cx > self.screen_width * 0.7 and cy < self.screen_height * 0.4:
                            logger.info("Detected close (×) icon via VisionAI – tapping …")
                            self._tap(cx, cy)
                            time.sleep(0.4)
                            if self.check_app_foreground():
                                logger.info("✓ Popup dismissed via VisionAI close button")
                                return True
                            # If tap caused focus loss, break to avoid further attempts
                            break
        except Exception as e:
            logger.debug(f"Close-button vision detection error: {e}")
        
        # Enhanced notification popup handling using OCR
        try:
            # Capture screenshot for OCR analysis
            screenshot_path = self.capture_screenshot("notification_detection")
            if screenshot_path:
                # Use VisionAI to analyze the screenshot
                elements = self.vision_engine.analyze(screenshot_path)
                
                # Look for notification-related keywords and buttons
                notification_keywords = [
                    "notification", "alert", "update", "new message", "reminder"
                ]
                action_keywords = [
                    "view", "open", "reply", "snooze", "dismiss", "clear", "yes", "no"
                ]
                
                is_notification_popup = False
                for element in elements:
                    element_text = element.text.lower() if hasattr(element, 'text') else ""
                    if any(keyword in element_text for keyword in notification_keywords):
                        is_notification_popup = True
                        break
                
                if is_notification_popup:
                    logger.info("Notification popup detected, analyzing for actions...")
                    for element in elements:
                        element_text = element.text.lower() if hasattr(element, 'text') else ""
                        
                        # Check if this looks like a notification action button
                        if any(keyword in element_text for keyword in action_keywords):
                            bbox = element.bbox.as_tuple() if hasattr(element, 'bbox') else []
                            if len(bbox) >= 4:
                                x = (bbox[0] + bbox[2]) // 2
                                y = (bbox[1] + bbox[3]) // 2
                                logger.info(f"Tapping notification action button via OCR: {element_text}")
                                self._tap(x, y)
                                time.sleep(0.5)
                                
                                if self.check_app_foreground():
                                    logger.info(f"✓ Notification popup handled with OCR: {element_text}")
                                    return True
                
                    logger.warning("No actionable buttons found on notification popup")
                else:
                    logger.info("No notification content detected via OCR")
            else:
                logger.warning("Could not capture screenshot for notification OCR analysis")
                
        except Exception as e:
            logger.error(f"Error during notification popup handling: {e}")
            
        return False
    
    def _try_dismiss_dialogs_recovery(self) -> bool:
        """Try to dismiss external dialogs and bring app to foreground."""
        logger.info("Attempting to dismiss external system dialogs and permissions...")
        
        # External popup indicators (system dialogs, permissions) - should be dismissed
        external_popup_indicators = {
            "system": ["google play", "android", "system", "settings", "permission"],
            "permission": ["allow", "deny", "permission", "access", "grant"],
            "notification": ["notification", "alert", "warning", "error", "update"],
            "external_app": ["chrome", "browser", "gmail", "maps", "play store"]
        }
        
        # Strategy 1: Dismiss external system popups (Google Play, Android system)
        external_popup_locations = [
            (500, 1200),  # Bottom center - "OK" or "Accept"
            (500, 1100),  # Slightly above bottom
            (500, 1300),  # Below bottom
            (300, 1200),  # Left side
            (700, 1200),  # Right side
        ]
        
        logger.info("Dismissing external system popups...")
        for x, y in external_popup_locations:
            self._tap(x, y)
            time.sleep(0.3)
        
        # Strategy 2: Dismiss permission dialogs (external - dismiss)
        permission_popup_locations = [
            (500, 1200),  # Bottom center - "Turn On" or "Allow"
            (500, 1100),  # Above bottom
            (500, 1300),  # Below bottom
            (400, 1200),  # Left side
            (600, 1200),  # Right side
            (500, 1000),  # Center area
        ]
        
        logger.info("Dismissing permission dialogs...")
        for x, y in permission_popup_locations:
            self._tap(x, y)
            time.sleep(0.3)
        
        # Strategy 3: Handle dismiss buttons (external - dismiss)
        dismiss_locations = [
            (500, 1200),  # Bottom center
            (500, 1100),  # Above bottom
            (500, 1300),  # Below bottom
            (300, 1200),  # Left side
            (700, 1200),  # Right side
            (500, 800),   # Center area
            (500, 900),   # Above center
        ]
        
        logger.info("Dismissing notification dialogs...")
        for x, y in dismiss_locations:
            self._tap(x, y)
            time.sleep(0.3)
        
        # Note: Internal popups (menu cards) are NOT dismissed here
        # They should be handled through the decision-making process
        logger.info("Internal popups (menu cards, navigation) will be handled through decision-making process")
        
        # Strategy 4: Look for Close/Dismiss buttons on external popups ONLY if we're certain it's external
        # This is more conservative to avoid dismissing internal menu cards
        logger.info("Checking if we should look for Close/Dismiss buttons...")
        
        # Only proceed if we're confident this is an external popup
        # For now, we'll be very conservative and avoid Close button dismissal
        # unless we have clear indicators of external system popups
        
        # Check if we have any clear external popup indicators
        current_foreground = self.get_current_foreground_app()
        has_external_indicators = any(indicator in current_foreground.lower() for indicator in 
                                    ["google", "android", "system", "settings", "permission", "chrome", "browser"])
        
        if has_external_indicators:
            logger.info("External popup detected, looking for Close/Dismiss buttons...")
            
            # Common Close/Dismiss button locations for external popups
            close_dismiss_buttons = [
                {"coords": (900, 200), "desc": "Close button (top right)"},
                {"coords": (800, 200), "desc": "Close button (top right area)"},
                {"coords": (500, 200), "desc": "Close button (top center)"},
                {"coords": (500, 1200), "desc": "Dismiss button (bottom center)"},
                {"coords": (500, 1100), "desc": "Dismiss button (above bottom)"},
                {"coords": (300, 1200), "desc": "Cancel button (bottom left)"},
                {"coords": (700, 1200), "desc": "Cancel button (bottom right)"},
            ]
            
            # Try each Close/Dismiss button location
            for button in close_dismiss_buttons:
                logger.info(f"Trying {button['desc']} on external popup")
                self._tap(*button['coords'])
                time.sleep(0.3)
                
                # Check if app is still in foreground after tap
                if self.check_app_foreground():
                    logger.info(f"Successfully used {button['desc']} to dismiss external popup")
                    break
                else:
                    logger.warning(f"App left foreground after {button['desc']}, stopping")
                    break
        else:
            logger.info("No clear external popup indicators found, skipping Close/Dismiss button attempts")
            logger.info("Internal popups (menu cards, navigation) should be handled through decision-making process")
        
        # Strategy 5: Try to bring app to front again
        subprocess.run(
            ["adb", "shell", "am", "start", "-n", f"{self.package_name}/.MainActivity"],
            capture_output=True,
            check=False
        )
        time.sleep(1)
        
        # Enhanced location prompt actions after dismiss_locations
        try:
            screen_info = subprocess.run(["adb", "shell", "wm", "size"], capture_output=True, text=True, check=False)
            if screen_info.stdout:
                size_match = re.search(r'(\d+)x(\d+)', screen_info.stdout)
                if size_match:
                    screen_width = int(size_match.group(1))
                    screen_height = int(size_match.group(2))
                    
                    center_x = screen_width // 2
                    bottom_y = int(screen_height * 0.85)
                    middle_y = int(screen_height * 0.7)
                    top_y = int(screen_height * 0.55)
                    
                    enhanced_location_actions = [
                        (center_x, bottom_y),      # Allow While Using the App
                        (center_x, middle_y),      # Allow Only While Using the App
                        (center_x, top_y),         # Allow All the Time
                        (center_x, bottom_y + 100), # Allow
                        (center_x, middle_y + 50),  # OK
                        (center_x, top_y - 50),     # Continue
                        (center_x - 100, bottom_y), # Left variant
                        (center_x + 100, bottom_y), # Right variant
                    ]
                    
                    for x, y in enhanced_location_actions:
                        self._tap(x, y)
                        time.sleep(0.5)
                        
                        # Check if permission dialog was dismissed
                        if self.check_app_foreground():
                            logger.info("✓ Permission dialog dismissed in recovery")
                            break
                else:
                    # Fallback coordinates
                    fallback_actions = [(500, 1200), (500, 1100), (500, 1300), (500, 1000), (500, 900), (500, 1000)]
                    for x, y in fallback_actions:
                        self._tap(x, y)
                        time.sleep(0.3)
            else:
                # Fallback coordinates
                fallback_actions = [(500, 1200), (500, 1100), (500, 1300), (500, 1000), (500, 900), (500, 1000)]
                for x, y in fallback_actions:
                    self._tap(x, y)
                    time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Error in enhanced location handling: {e}")
            # Fallback coordinates
            fallback_actions = [(500, 1200), (500, 1100), (500, 1300), (500, 1000), (500, 900), (500, 1000)]
            for x, y in fallback_actions:
                self._tap(x, y)
                time.sleep(0.3)
        
        return True
    
    def _try_recent_apps_recovery(self) -> bool:
        """Try to recover app using recent apps."""
        subprocess.run(["adb", "shell", "input", "keyevent", "187"], capture_output=True, check=False)  # Recent apps
        time.sleep(1)
        subprocess.run(["adb", "shell", "input", "tap", "400", "800"], capture_output=True, check=False)  # Tap app
        time.sleep(1)
        return True
    
    def _try_home_relaunch_recovery(self) -> bool:
        """Try to recover app by going home and relaunching."""
        subprocess.run(["adb", "shell", "input", "keyevent", "3"], capture_output=True, check=False)  # Home
        time.sleep(1)
        subprocess.run(["adb", "shell", "monkey", "-p", self.package_name, "-c", "android.intent.category.LAUNCHER", "1"], capture_output=True, check=False)
        time.sleep(2)
        return True
    
    def _try_force_restart_recovery(self) -> bool:
        """Try to recover app by force stopping and restarting."""
        subprocess.run(["adb", "shell", "am", "force-stop", self.package_name], capture_output=True, check=False)
        time.sleep(1)
        subprocess.run(["adb", "shell", "monkey", "-p", self.package_name, "-c", "android.intent.category.LAUNCHER", "1"], capture_output=True, check=False)
        time.sleep(2)
        return True
    
    def _try_clear_recents_recovery(self) -> bool:
        """Try to recover app by clearing recent apps and relaunching."""
        subprocess.run(["adb", "shell", "input", "keyevent", "187"], capture_output=True, check=False)  # Recent apps
        time.sleep(1)
        subprocess.run(["adb", "shell", "input", "keyevent", "4"], capture_output=True, check=False)  # Back to clear
        time.sleep(1)
        subprocess.run(["adb", "shell", "monkey", "-p", self.package_name, "-c", "android.intent.category.LAUNCHER", "1"], capture_output=True, check=False)
        time.sleep(2)
        return True
    
    def _try_direct_app_launch_recovery(self) -> bool:
        """Try to recover app using direct app launch - single strategy approach."""
        logger.info(f"Attempting direct app launch recovery for {self.package_name}")
        
        try:
            # Enhanced activity list with more variations
            common_activities = [
                f"{self.package_name}/.MainActivity",
                f"{self.package_name}/.ui.MainActivity", 
                f"{self.package_name}/.activities.MainActivity",
                f"{self.package_name}/.LauncherActivity",
                f"{self.package_name}/.SplashActivity",
                f"{self.package_name}/.StartupActivity",
                f"{self.package_name}/.HomeActivity",
                f"{self.package_name}/.MenuActivity",
                f"{self.package_name}/.activity.MainActivity",
                f"{self.package_name}/.activity.LauncherActivity",
                f"{self.package_name}/.activity.SplashActivity",
                f"{self.package_name}/.activity.StartupActivity",
                f"{self.package_name}/.activity.HomeActivity",
                f"{self.package_name}/.activity.MenuActivity"
            ]
            
            # Try common activities first (fast path)
            for activity in common_activities:
                try:
                    logger.info(f"Trying to launch: {activity}")
                    result = subprocess.run(["adb", "shell", "am", "start", "-n", activity], 
                                          capture_output=True, text=True, check=False)
                    
                    # Log the result for debugging
                    if result.returncode != 0:
                        logger.debug(f"Launch failed for {activity}: {result.stderr}")
                    else:
                        logger.debug(f"Launch command succeeded for {activity}")
                    
                    # Quick check after a short delay
                    time.sleep(0.5)  # Increased delay for better reliability
                    
                    if self.check_app_foreground():
                        logger.info(f"✓ Successfully launched app with: {activity}")
                        return True
                    else:
                        logger.debug(f"App not in foreground after launching: {activity}")
                        
                except Exception as e:
                    logger.debug(f"Exception launching {activity}: {e}")
                    continue
            
            # If common activities failed, try activity discovery (slower but more thorough)
            logger.info("Common activities failed, trying activity discovery...")
            main_activity = self._discover_main_activity()
            
            if main_activity:
                logger.info(f"Launching app with discovered activity: {main_activity}")
                result = subprocess.run(["adb", "shell", "am", "start", "-n", main_activity], 
                                      capture_output=True, text=True, check=False)
                
                if result.returncode != 0:
                    logger.warning(f"Discovered activity launch failed: {result.stderr}")
                
                time.sleep(0.5)
                
                if self.check_app_foreground():
                    logger.info(f"✓ Successfully launched app with discovered activity: {main_activity}")
                    return True
                else:
                    logger.warning("App not in foreground after discovered activity launch")
            else:
                logger.warning("No main activity discovered")
            
            # Try monkey launcher as last resort
            logger.info("Trying monkey launcher as last resort...")
            try:
                subprocess.run(["adb", "shell", "monkey", "-p", self.package_name, "-c", "android.intent.category.LAUNCHER", "1"], 
                              capture_output=True, check=False)
                time.sleep(1)
                
                if self.check_app_foreground():
                    logger.info("✓ Successfully launched app with monkey launcher")
                    return True
                else:
                    logger.warning("App not in foreground after monkey launcher")
            except Exception as e:
                logger.debug(f"Monkey launcher failed: {e}")
            
            # Final check
            time.sleep(0.5)
            final_check = self.check_app_foreground()
            if final_check:
                logger.info("✓ App is in foreground after recovery attempts")
                return True
            else:
                logger.error("❌ App still not in foreground after all recovery attempts")
                return False
            
        except Exception as e:
            logger.error(f"Direct app launch recovery failed with exception: {e}")
            return False
    
    def _discover_main_activity(self) -> str | None:
        """Discover the main activity for the target package."""
        try:
            # Use cmd package resolve-activity to find the main activity
            result = subprocess.run(
                ["adb", "shell", "cmd", "package", "resolve-activity", "--brief", self.package_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if self.package_name in line and "activity" in line:
                        # Extract the activity name from the output
                        # Format: package_name/activity_name
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            activity_name = parts[-1]  # Last part is usually the activity
                            logger.info(f"Discovered main activity: {activity_name}")
                            return activity_name
            
            # Alternative: use dumpsys to find activities
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "package", self.package_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "android.intent.action.MAIN" in line and "android.intent.category.LAUNCHER" in line:
                        # Extract activity name from the line
                        if self.package_name in line:
                            # Look for the activity name pattern
                            import re
                            match = re.search(rf'{self.package_name}/([^\s]+)', line)
                            if match:
                                activity_name = f"{self.package_name}/{match.group(1)}"
                                logger.info(f"Discovered launcher activity: {activity_name}")
                                return activity_name
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to discover main activity: {e}")
            return None
    
    def check_app_foreground(self) -> bool:
        """Check if the app is currently in foreground with proper detection."""
        try:
            # Method 1: Check current activity using dumpsys activity activities
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "activity", "activities"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Look for ResumedActivity in the output (without 'm' prefix)
                for line in result.stdout.split('\n'):
                    if 'ResumedActivity:' in line and self.package_name in line:
                        logger.info(f"✓ App is in foreground (ResumedActivity: {line.strip()})")
                        return True
            
            # Method 2: Check current focus using dumpsys window windows
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "window", "windows"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Look for mCurrentFocus in the output
                for line in result.stdout.split('\n'):
                    if 'mCurrentFocus' in line and self.package_name in line:
                        logger.info(f"✓ App is in foreground (mCurrentFocus: {line.strip()})")
                        return True
            
            # Method 3: Check top activity (but only if it's actually resumed)
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "activity", "top"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Look for ACTIVITY in the output, but be more careful
                for line in result.stdout.split('\n'):
                    if 'ACTIVITY' in line and self.package_name in line:
                        # Additional check: make sure it's not just a background activity
                        if 'state=RESUMED' in line or 'visible=true' in line:
                            logger.info(f"✓ App is in foreground (top activity: {line.strip()})")
                            return True
                        else:
                            logger.debug(f"Found app in top activity but not resumed: {line.strip()}")
            
            # Method 5: Check using dumpsys window | grep -E "mCurrentFocus.*{package_name}"
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "window"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Use grep-like filtering in Python
                import re
                pattern = rf"mCurrentFocus.*{re.escape(self.package_name)}"
                for line in result.stdout.split('\n'):
                    if re.search(pattern, line):
                        logger.info(f"✓ App is in foreground (window focus match: {line.strip()})")
                        return True
            
            # If we get here, the app is not in foreground
            logger.warning(f"❌ App {self.package_name} is NOT in foreground")
            
            # Log what IS in foreground for debugging
            try:
                # Get current foreground app for debugging
                result = subprocess.run(
                    ["adb", "shell", "dumpsys", "activity", "activities"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'ResumedActivity:' in line:
                            logger.info(f"Current foreground app: {line.strip()}")
                            break
            except Exception as e:
                logger.debug(f"Could not get current foreground app info: {e}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking app foreground: {e}")
            return False
    
    def uninstall_app(self) -> bool:
        """Uninstall the app after testing.
        
        Returns
        -------
        bool
            True if uninstallation successful

        """
        logger.info(f"Uninstalling {self.app_name}...")
        
        try:
            result = subprocess.run(
                ["adb", "uninstall", self.package_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            if "Success" in result.stdout:
                logger.info("✓ App uninstalled successfully")
                return True
            else:
                logger.warning(f"Uninstallation may have failed: {result.stdout}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Uninstallation error: {e}")
            return False
    
    def cleanup_after_testing(self) -> None:
        """Clean up after testing is complete."""
        logger.info("Performing cleanup...")
        try:
            # Restore original device settings
            subprocess.run(["adb", "shell", "settings", "put", "global", "window_animation_scale", "1.0"], capture_output=True, check=False)
            subprocess.run(["adb", "shell", "settings", "put", "global", "transition_animation_scale", "1.0"], capture_output=True, check=False)
            subprocess.run(["adb", "shell", "settings", "put", "global", "animator_duration_scale", "1.0"], capture_output=True, check=False)
            subprocess.run(["adb", "shell", "settings", "put", "global", "stay_on_while_plugged_in", "0"], capture_output=True, check=False)
            subprocess.run(["adb", "shell", "settings", "put", "system", "screen_brightness_mode", "1"], capture_output=True, check=False)
            
            # Re-enable system UI overlays
            subprocess.run(["adb", "shell", "settings", "put", "secure", "overlay_enabled", "1"], capture_output=True, check=False)
            
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def capture_screenshot(self, action_name: str) -> Optional[str]:
        """Capture a screenshot and save it.
        
        Parameters
        ----------
        action_name : str
            Name for the screenshot file
            
        Returns
        -------
        Optional[str]
            Path to saved screenshot, or None if failed

        """
        try:
            timestamp = int(time.time() * 1000)
            filename = f"{self.app_name.lower().replace(' ', '_')}_{action_name}_{timestamp}.png"
            screenshot_path = self.screenshots_dir / filename
            
            # Capture screenshot
            result = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                capture_output=True,
                check=True
            )
            
            # Save screenshot
            with open(screenshot_path, 'wb') as f:
                f.write(result.stdout)
            
            logger.info(f"✓ Screenshot saved: {filename}")
            return str(screenshot_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def should_capture_screenshot(self, action_type: str, action_step: str = "after") -> bool:
        """Determine if a screenshot should be captured based on frequency settings.
        
        Parameters
        ----------
        action_type : str
            Type of action being performed
        action_step : str
            Step in action: "before", "during", "after"
            
        Returns
        -------
        bool
            True if screenshot should be captured

        """
        if self.screenshot_frequency == "minimal":
            # Only capture after action completion
            return action_step == "after"
        
        elif self.screenshot_frequency == "high":
            # Capture at every step
            return True
        
        elif self.screenshot_frequency == "adaptive":
            # Smart capture based on action type and step
            if action_step == "after":
                return True  # Always capture after action
            
            elif action_step == "before":
                # Capture before complex actions
                complex_actions = ["swipe", "keyevent"]
                return action_type in complex_actions
            
            elif action_step == "during":
                # Capture during long actions (swipes)
                return action_type == "swipe"
        
        return False
    
    def capture_screenshot_with_timing(self, action_name: str, action_step: str = "after") -> Optional[str]:
        """Capture a screenshot with timing information.
        
        Parameters
        ----------
        action_name : str
            Name for the screenshot file
        action_step : str
            Step in action: "before", "during", "after"
            
        Returns
        -------
        Optional[str]
            Path to saved screenshot, or None if failed

        """
        try:
            timestamp = int(time.time() * 1000)
            filename = f"{self.app_name.lower().replace(' ', '_')}_{action_name}_{action_step}_{timestamp}.png"
            screenshot_path = self.screenshots_dir / filename
            
            # Capture screenshot
            result = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                capture_output=True,
                check=True
            )
            
            # Save screenshot
            with open(screenshot_path, 'wb') as f:
                f.write(result.stdout)
            
            logger.info(f"✓ Screenshot saved: {filename}")
            return str(screenshot_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def capture_screenshot_optimized(self, action_name: str, action_step: str = "after", 
                                   immediate: bool = True) -> Optional[str]:
        """Capture screenshot with optimized timing to avoid foreground issues.
        
        Parameters
        ----------
        action_name : str
            Name for the screenshot file
        action_step : str
            Step in action: "before", "during", "after"
        immediate : bool
            If True, capture immediately without foreground checks
            
        Returns
        -------
        Optional[str]
            Path to saved screenshot, or None if failed

        """
        try:
            timestamp = int(time.time() * 1000)
            filename = f"{self.app_name.lower().replace(' ', '_')}_{action_name}_{action_step}_{timestamp}.png"
            screenshot_path = self.screenshots_dir / filename
            
            # OPTIMIZATION: Capture immediately after action, before any system interference
            if immediate:
                # Use fast screenshot capture without foreground checks
                result = subprocess.run(
                    ["adb", "exec-out", "screencap", "-p"],
                    capture_output=True,
                    check=True,
                    timeout=5  # Add timeout to prevent hanging
                )
            else:
                # Traditional approach with foreground checks
                if not self.check_app_foreground():
                    self.keep_app_foreground()
                    time.sleep(0.5)  # Brief wait
                
                result = subprocess.run(
                    ["adb", "exec-out", "screencap", "-p"],
                    capture_output=True,
                    check=True
                )
            
            # Save screenshot
            with open(screenshot_path, 'wb') as f:
                f.write(result.stdout)
            
            # Track last screenshot path for Phi Ground
            self._last_screenshot_path = str(screenshot_path)
            
            logger.info(f"✓ Screenshot saved successfully: {filename}")
            return str(screenshot_path)
            
        except subprocess.TimeoutExpired:
            logger.error(f"Screenshot capture timed out: {action_name}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected screenshot error: {e}")
            return None
    
    def capture_screenshot_with_foreground_management(self, action_name: str, action_step: str = "after", 
                                                    max_retries: int = 3) -> Optional[str]:
        """Capture screenshot with optimized timing for frequent actions.
        
        Parameters
        ----------
        action_name : str
            Name for the screenshot file
        action_step : str
            Step in action: "before", "during", "after"
        max_retries : int
            Maximum number of retry attempts
            
        Returns
        -------
        Optional[str]
            Path to saved screenshot, or None if failed

        """
        for attempt in range(max_retries):
            try:
                timestamp = int(time.time() * 1000)
                filename = f"{self.app_name.lower().replace(' ', '_')}_{action_name}_{action_step}_{timestamp}.png"
                screenshot_path = self.screenshots_dir / filename
                
                # ENHANCED FOREGROUND MANAGEMENT: Check and recover if needed
                if attempt == 0:  # Only check foreground on first attempt
                    is_foreground = self.check_app_foreground()
                    if not is_foreground:
                        logger.warning("App not in foreground, attempting enhanced recovery...")
                        if not self.enhanced_foreground_recovery():
                            logger.error("Failed to recover app to foreground")
                            return None
                        time.sleep(0.5)  # Wait for recovery to stabilize
                
                # OPTIMIZATION 2: Minimal wait time for app stabilization
                wait_time = 0.1 + (attempt * 0.1)  # Progressive wait: 0.1s, 0.2s, 0.3s (reduced from 0.5s, 0.8s, 1.1s)
                if wait_time > 0.1:
                    logger.info(f"Waiting {wait_time:.1f}s for app to stabilize...")
                    time.sleep(wait_time)
                
                # OPTIMIZATION 3: Capture screenshot immediately with timeout
                logger.info("Capturing screenshot...")
                result = subprocess.run(
                    ["adb", "exec-out", "screencap", "-p"],
                    capture_output=True,
                    check=True,
                    timeout=3  # Reduced from 5s
                )
                
                # OPTIMIZATION 4: Save screenshot
                with open(screenshot_path, 'wb') as f:
                    f.write(result.stdout)
                
                # Track last screenshot path for Phi Ground
                self._last_screenshot_path = str(screenshot_path)
                
                logger.info(f"✓ Screenshot saved successfully: {filename}")
                return str(screenshot_path)
                
            except subprocess.TimeoutExpired:
                logger.error(f"Screenshot capture timed out (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Reduced from 1s
                    continue
                else:
                    return None
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Screenshot capture failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Reduced from 1s
                    continue
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected screenshot error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Reduced from 1s
                    continue
                else:
                    return None
        
        return None
    
    def perform_random_action_with_optimized_timing(self) -> Tuple[str, str, str, List[str]]:
        """Perform a random action with optimized screenshot timing to avoid foreground issues.
        
        Returns
        -------
        Tuple[str, str, str, List[str]]
            (action_type, action_description, element_id, screenshot_paths)

        """
        actions = [
            ("tap", "Tap center of screen", "center_tap"),
            ("tap", "Tap top-left area", "top_left_tap"),
            ("tap", "Tap top-right area", "top_right_tap"),
            ("tap", "Tap bottom-left area", "bottom_left_tap"),
            ("tap", "Tap bottom-right area", "bottom_right_tap"),
            ("swipe", "Swipe down", "swipe_down"),
            ("swipe", "Swipe up", "swipe_up"),
            ("swipe", "Swipe left", "swipe_left"),
            ("swipe", "Swipe right", "swipe_right"),
            ("keyevent", "Press back button", "back_button"),
            # ("keyevent", "Press home button", "home_button"),  # BLOCKED: Takes app out of foreground
            # ("keyevent", "Press menu button", "menu_button"),  # BLOCKED: Can cause navigation issues
        ]
        
        action_type, description, element_id = random.choice(actions)
        screenshot_paths = []
        
        try:
            # STRATEGY 1: Capture before action (if needed) with immediate timing
            if self.should_capture_screenshot(action_type, "before"):
                before_screenshot = self.capture_screenshot_optimized(
                    f"action_{self.action_count + 1}", "before", immediate=True
                )
                if before_screenshot:
                    screenshot_paths.append(before_screenshot)
            
            # STRATEGY 2: Perform action with minimal delay
            if action_type == "tap":
                # Try precision coordinates first
                x_p, y_p = self._get_precision_tap_coordinates()
                if x_p is None or y_p is None:
                    x_p = random.randint(100, 900)
                    y_p = random.randint(200, 1600)
                self._tap(x_p, y_p)
                # CRITICAL: Capture immediately after tap, before any system interference
                time.sleep(0.2)  # Minimal wait for tap to register
                
            elif action_type == "swipe":
                # Swipe with optimized timing
                start_x = random.randint(100, 900)
                start_y = random.randint(200, 1600)
                end_x = random.randint(100, 900)
                end_y = random.randint(200, 1600)
                
                # Execute swipe
                subprocess.run(
                    ["adb", "shell", "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y)],
                    capture_output=True,
                    check=False
                )
                
                # STRATEGY 3: Capture during swipe at optimal moment
                if self.should_capture_screenshot(action_type, "during"):
                    time.sleep(0.3)  # Wait for swipe to be in progress
                    during_screenshot = self.capture_screenshot_optimized(
                        f"action_{self.action_count + 1}", "during", immediate=True
                    )
                    if during_screenshot:
                        screenshot_paths.append(during_screenshot)
                
                # Wait for swipe completion
                time.sleep(0.7)  # Reduced wait time
                
            elif action_type == "keyevent":
                # Key events with immediate capture
                key_map = {
                    "back_button": "4",
                    "home_button": "3",
                    "menu_button": "82"
                }
                key_code = key_map.get(element_id, "4")
                subprocess.run(
                    ["adb", "shell", "input", "keyevent", key_code],
                    capture_output=True,
                    check=False
                )
                time.sleep(0.3)  # Brief wait for key event
            
            # STRATEGY 4: Capture after action with immediate timing
            if self.should_capture_screenshot(action_type, "after"):
                after_screenshot = self.capture_screenshot_optimized(
                    f"action_{self.action_count + 1}", "after", immediate=True
                )
                if after_screenshot:
                    screenshot_paths.append(after_screenshot)
            
        except Exception as e:
            logger.warning(f"Action failed: {e}")
        
        return action_type, description, element_id, screenshot_paths
    
    def perform_random_action_with_foreground_management(self) -> Tuple[str, str, str, List[str]]:
        """Perform a random action with optimized timing for frequent meaningful transitions.
        
        Returns
        -------
        Tuple[str, str, str, List[str]]
            (action_type, action_description, element_id, screenshot_paths)

        """
        # Try Phi Ground first if available
        phi_ground_action = self._try_phi_ground_action()
        if phi_ground_action:
            logger.info("Using Phi Ground generated action")
            return phi_ground_action
        
        # Fallback to random actions if Phi Ground is not available
        logger.info("Using random action (Phi Ground not available)")
        
        # Enhanced action set with more meaningful interactions
        actions = [
            # High-impact actions that create meaningful state changes
            ("tap", "Tap primary action button", "primary_action"),
            ("tap", "Tap secondary action button", "secondary_action"),
            ("tap", "Tap navigation element", "navigation_tap"),
            ("tap", "Tap interactive control", "interactive_control"),
            ("swipe", "Swipe to next screen", "swipe_next"),
            ("swipe", "Swipe to previous screen", "swipe_prev"),
            ("swipe", "Swipe to reveal menu", "swipe_menu"),
            ("swipe", "Swipe to scroll content", "swipe_scroll"),
            ("keyevent", "Press back button", "back_button"),
            # ("keyevent", "Press home button", "home_button"),  # BLOCKED: Takes app out of foreground
            # ("keyevent", "Press menu button", "menu_button"),  # BLOCKED: Can cause navigation issues
            # Quick exploratory actions
            ("tap", "Tap center area", "center_tap"),
            ("tap", "Tap edge area", "edge_tap"),
            ("swipe", "Quick swipe gesture", "quick_swipe"),
        ]
        
        action_type, description, element_id = random.choice(actions)
        screenshot_paths = []
        
        try:
            # ENHANCED FOREGROUND MANAGEMENT: Check and recover if needed
            if not self.check_app_foreground():
                logger.warning("App not in foreground, attempting enhanced recovery...")
                if not self.enhanced_foreground_recovery():
                    logger.error("Failed to recover app to foreground")
                    return action_type, description, element_id, screenshot_paths
                time.sleep(0.5)  # Wait for recovery to stabilize
            
            # OPTIMIZATION 2: Capture before screenshot only for high-impact actions
            if (action_type in ["tap"] and "primary" in description) or "navigation" in description:
                before_screenshot = self.capture_screenshot_optimized(
                    f"action_{self.action_count + 1}", "before", immediate=True
                )
                if before_screenshot:
                    screenshot_paths.append(before_screenshot)
            
            # OPTIMIZATION 3: Perform action with minimal delays
            logger.info(f"Performing {action_type}: {description}")
            
            if action_type == "tap":
                # Intelligent tap positioning for meaningful interactions
                if "primary" in description:
                    # Tap in likely primary action areas
                    x = random.choice([400, 500, 600])  # Center area
                    y = random.choice([1200, 1300, 1400])  # Bottom area
                elif "navigation" in description:
                    # Tap in navigation areas
                    x = random.choice([100, 1200])  # Left or right edge
                    y = random.choice([200, 300, 400])  # Top area
                elif "interactive" in description:
                    # Tap in interactive areas
                    x = random.randint(200, 800)
                    y = random.randint(400, 1200)
                else:
                    # General tap
                    x = random.randint(100, 900)
                    y = random.randint(200, 1600)
                
                self._tap(x, y)
                time.sleep(0.15)  # Reduced from 0.3s - just enough for tap to register
                
                # Attempt autofill if a keyboard popped up
                self._maybe_fill_text_field()
            
            elif action_type == "swipe":
                # Intelligent swipe patterns for meaningful transitions
                if "next" in description:
                    # Swipe left to right (next screen)
                    start_x, end_x = 100, 800
                    start_y = end_y = random.randint(800, 1200)
                elif "prev" in description:
                    # Swipe right to left (previous screen)
                    start_x, end_x = 800, 100
                    start_y = end_y = random.randint(800, 1200)
                elif "menu" in description:
                    # Swipe from edge to reveal menu
                    start_x, end_x = 50, 300
                    start_y = end_y = random.randint(400, 1000)
                elif "scroll" in description:
                    # Vertical scroll
                    start_x = end_x = random.randint(300, 700)
                    start_y, end_y = 1200, 400
                else:
                    # General swipe
                    start_x = random.randint(100, 900)
                    start_y = random.randint(200, 1600)
                    end_x = random.randint(100, 900)
                    end_y = random.randint(200, 1600)
                
                subprocess.run(
                    ["adb", "shell", "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y)],
                    capture_output=True,
                    check=False
                )
                
                # OPTIMIZATION 4: Capture during swipe only for long swipes
                if "next" in description or "prev" in description:
                    time.sleep(0.2)  # Reduced from 0.4s
                    during_screenshot = self.capture_screenshot_optimized(
                        f"action_{self.action_count + 1}", "during", immediate=True
                    )
                    if during_screenshot:
                        screenshot_paths.append(during_screenshot)
                
                time.sleep(0.4)  # Reduced from 0.8s - just enough for swipe to complete
                
                # Attempt autofill if a keyboard popped up
                self._maybe_fill_text_field()
            
            elif action_type == "keyevent":
                # Key events with enhanced recovery for back button
                key_map = {
                    "back_button": "4",
                    "home_button": "3",
                    "menu_button": "82"
                }
                key_code = key_map.get(element_id, "4")
                subprocess.run(
                    ["adb", "shell", "input", "keyevent", key_code],
                    capture_output=True,
                    check=False
                )
                
                # Enhanced recovery for back button actions
                if element_id == "back_button":
                    time.sleep(1)  # Wait longer for back button effects
                    logger.info("Enhanced recovery check after back button action...")
                    
                    # Check if app is still in foreground
                    if not self.check_app_foreground():
                        logger.warning("Back button may have taken app out of foreground, attempting recovery...")
                        
                        # Try enhanced recovery
                        if self.enhanced_foreground_recovery():
                            time.sleep(1)
                            if self.check_app_foreground():
                                logger.info("✓ App recovered after back button action")
                            else:
                                logger.error("❌ Failed to recover app after back button action")
                        else:
                            logger.error("❌ Enhanced recovery failed after back button action")
                else:
                    time.sleep(0.2)  # Normal delay for other key events
                
                # Attempt autofill if a keyboard popped up
                self._maybe_fill_text_field()
            
            # OPTIMIZATION 5: Capture after screenshot with minimal delay
            after_screenshot = self.capture_screenshot_optimized(
                f"action_{self.action_count + 1}", "after", immediate=True
            )
            if after_screenshot:
                screenshot_paths.append(after_screenshot)
            
            # Attempt autofill if a keyboard popped up
            self._maybe_fill_text_field()
        
        except Exception as e:
            logger.warning(f"Action failed: {e}")
        
        return action_type, description, element_id, screenshot_paths
    
    def _try_phi_ground_action(self) -> Optional[Tuple[str, str, str, List[str]]]:
        """Try to generate action using Phi Ground.
        
        Returns
        -------
        Optional[Tuple[str, str, str, List[str]]]
            (action_type, action_description, element_id, screenshot_paths) or None if Phi Ground fails
        """
        try:
            # Check if Phi Ground is enabled
            from src.core.config import config
            if not config.use_phi_ground:
                return None
            
            # Check if we have a recent screenshot
            if not hasattr(self, '_last_screenshot_path') or not self._last_screenshot_path:
                return None
            
            # Import Phi Ground
            from src.ai.phi_ground import get_phi_ground_generator
            from src.vision.models import UIElement, BoundingBox
            
            # Get Phi Ground generator
            phi_ground = get_phi_ground_generator()
            
            # Create sample UI elements (in a real scenario, these would come from vision analysis)
            # For now, we'll create some basic elements based on common UI patterns
            ui_elements = [
                UIElement(
                    bbox=BoundingBox(100, 200, 300, 250),
                    text="Login",
                    confidence=0.9,
                    element_type="button"
                ),
                UIElement(
                    bbox=BoundingBox(100, 300, 400, 350),
                    text="Email",
                    confidence=0.8,
                    element_type="input"
                ),
                UIElement(
                    bbox=BoundingBox(100, 400, 400, 450),
                    text="Password",
                    confidence=0.8,
                    element_type="input"
                )
            ]
            
            # Task description based on current app state
            task_description = f"Explore and interact with {self.app_name} app"
            
            # Action history (simplified)
            action_history = []
            
            # Generate action using Phi Ground
            import asyncio
            action = asyncio.run(phi_ground.generate_touch_action(
                self._last_screenshot_path, task_description, action_history, ui_elements
            ))
            
            if action:
                action_type = action.get("type", "tap")
                reasoning = action.get("reasoning", "Phi Ground generated")
                element_id = f"phi_ground_{action_type}"
                
                # Convert to the expected format
                return action_type, reasoning, element_id, []
            
            return None
            
        except Exception as e:
            logger.warning(f"Phi Ground action generation failed: {e}")
            return None
    
    def analyze_screenshot_with_optimized_timing(self, screenshot_path: str) -> List[Dict]:
        """Analyze screenshot with optimized timing to minimize foreground issues.
        
        Parameters
        ----------
        screenshot_path : str
            Path to screenshot
            
        Returns
        -------
        List[Dict]
            List of detected elements

        """
        try:
            # STRATEGY: Analyze immediately after capture, before any system interference
            start_time = time.time()
            
            # Perform VisionAI analysis immediately (no foreground checks during analysis)
            elements = self.vision_engine.analyze(screenshot_path)
            
            analysis_time = time.time() - start_time
            
            # Only check foreground after analysis is complete (but don't fail if it's not in foreground)
            if not self.check_app_foreground():
                logger.debug(f"App not in foreground after VisionAI analysis (took {analysis_time:.2f}s)")
                # Don't try to recover here - let the main action loop handle foreground management
                
            # Log performance
            if analysis_time > 1.0:
                logger.info(f"VisionAI analysis took {analysis_time:.2f}s")
            
            return [
                {
                    "text": element.text,
                    "confidence": element.confidence,
                    "element_type": element.element_type,
                    "bbox": element.bbox.as_tuple()
                }
                for element in elements
            ]
        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")
            return []
    
    def analyze_screenshot_with_foreground_management(self, screenshot_path: str) -> List[Dict]:
        """Analyze screenshot with optimized timing for frequent actions.
        
        Parameters
        ----------
        screenshot_path : str
            Path to screenshot
            
        Returns
        -------
        List[Dict]
            List of detected elements

        """
        try:
            # OPTIMIZATION 1: Skip foreground check for analysis (already done before capture)
            start_time = time.time()
            
            # OPTIMIZATION 2: Perform VisionAI analysis immediately
            if self.vision_engine is None:
                logger.warning("Vision engine not initialized, skipping analysis")
                return []
                
            elements = self.vision_engine.analyze(screenshot_path)
            analysis_time = time.time() - start_time
            
            # OPTIMIZATION 3: Log performance only if analysis is slow
            if analysis_time > 0.5:  # Reduced threshold from 1.0s
                logger.info(f"VisionAI analysis took {analysis_time:.2f}s")
            
            return [
                {
                    "text": element.text,
                    "confidence": element.confidence,
                    "element_type": element.element_type,
                    "bbox": element.bbox.as_tuple()
                }
                for element in elements
            ]
        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")
            return []
    
    def create_llm_event(self, action_type: str, description: str, element_id: str, 
                        screenshot_path: str, reasoning: str = "") -> LLMEvent:
        """Create an LLM event for the action.
        
        Parameters
        ----------
        action_type : str
            Type of action performed
        description : str
            Description of the action
        element_id : str
            ID of the element interacted with
        screenshot_path : str
            Path to the screenshot
        reasoning : str
            Reasoning for the action
            
        Returns
        -------
        LLMEvent
            Created LLM event

        """
        next_state_id = f"state_{self.action_count + 1}"
        
        event = LLMEvent(
            timestamp=datetime.now().isoformat(),
            state_id=self.current_state_id,
            element_id=element_id,
            action=action_type,
            reasoning=reasoning or f"Performed {description} during automated testing",
            resulting_state_id=next_state_id,
            screenshot_path=screenshot_path
        )
        
        # Persist reasoning to a plain-text log for external review
        try:
            llm_log = self.output_dir / "LLM_logs.txt"
            with open(llm_log, "a", encoding="utf-8") as fp:
                fp.write(f"[{event.timestamp}] {event.action} -> {event.reasoning}\n")
        except Exception as log_err:
            logger.debug(f"Could not write LLM log: {log_err}")
        
        return event
    
    def create_app_state(self, state_id: str, screenshot_path: str, elements: List[Dict]) -> AppState:
        """Create an app state.
        
        Parameters
        ----------
        state_id : str
            State ID
        screenshot_path : str
            Path to screenshot
        elements : List[Dict]
            List of detected elements
            
        Returns
        -------
        AppState
            Created app state

        """
        state = AppState(
            state_id=state_id,
            activity_name=f"{self.package_name}.MainActivity",
            description=f"App state after action {self.action_count}",
            screenshot_path=screenshot_path,
            elements=elements
        )
        
        return state
    
    def optimize_device_for_automation(self) -> None:
        """Optimize device settings to reduce app focus loss during automation."""
        logger.info("Optimizing device settings for automation...")
        
        try:
            # Disable animations for faster and more stable automation
            subprocess.run(
                ["adb", "shell", "settings", "put", "global", "window_animation_scale", "0.0"],
                capture_output=True,
                check=False
            )
            subprocess.run(
                ["adb", "shell", "settings", "put", "global", "transition_animation_scale", "0.0"],
                capture_output=True,
                check=False
            )
            subprocess.run(
                ["adb", "shell", "settings", "put", "global", "animator_duration_scale", "0.0"],
                capture_output=True,
                check=False
            )
            
            # Keep screen awake during automation
            subprocess.run(
                ["adb", "shell", "settings", "put", "global", "stay_on_while_plugged_in", "3"],
                capture_output=True,
                check=False
            )
            
            # Disable auto-brightness for consistent screenshots
            subprocess.run(
                ["adb", "shell", "settings", "put", "system", "screen_brightness_mode", "0"],
                capture_output=True,
                check=False
            )
            
            # Set fixed brightness
            subprocess.run(
                ["adb", "shell", "settings", "put", "system", "screen_brightness", "128"],
                capture_output=True,
                check=False
            )
            
            # Disable system UI overlays that might interfere
            subprocess.run(
                ["adb", "shell", "settings", "put", "secure", "overlay_enabled", "0"],
                capture_output=True,
                check=False
            )
            
            logger.info("✓ Device settings optimized for automation")
            
        except Exception as e:
            logger.warning(f"Device optimization warning: {e}")
    
    def run_testing_pipeline(self) -> bool:
        """Run the complete testing pipeline.
        
        Returns
        -------
        bool
            True if testing completed successfully

        """
        logger.info("=" * 70)
        logger.info(f"STARTING UNIVERSAL APK TESTING: {self.app_name}")
        logger.info("=" * 70)
        
        # Record test start time
        self.test_start_time = datetime.now()
        logger.info(f"Test started at: {self.test_start_time}")
        
        try:
            # Step 1: Environment check
            if not self.check_environment():
                return False
            
            # Step 1.5: Optimize device settings
            self.optimize_device_for_automation()
            
            # Step 2: Install APK
            if not self.install_apk():
                return False
            
            # Step 3: Launch app
            if not self.launch_app():
                return False
            
            # Step 4: Wait for app to fully stabilize and ensure it's in foreground
            logger.info("Step 4: Ensuring app is fully launched and in foreground...")
            time.sleep(2)  # Additional stabilization time
            
            # Try to bring app to foreground if needed (after launch sequence)
            if not self.check_app_foreground():
                logger.info("App not in foreground after launch, attempting to bring to front...")
                if not self.enhanced_foreground_recovery():
                    logger.warning("Failed to bring app to foreground after launch")
                else:
                    logger.info("✓ App successfully brought to foreground after launch")
            
            # Step 5: Capture initial screenshot (only after app is stable)
            logger.info("Step 5: Capturing initial screenshot...")
            initial_screenshot = self.capture_screenshot("initial")
            if initial_screenshot:
                # Analyze initial state
                elements = self.analyze_screenshot_with_optimized_timing(initial_screenshot)
                initial_state = self.create_app_state("state_0", initial_screenshot, elements)
                self.app_states["state_0"] = initial_state
            
            # Step 6: Perform actions and capture screenshots
            logger.info(f"Performing {self.num_actions} actions...")
            
            for action_num in range(1, self.num_actions + 1):
                logger.info(f"Action {action_num}/{self.num_actions}")
                
                # Ensure app is in foreground before each action (only after launch sequence is complete)
                if not self.check_app_foreground():
                    logger.info("App not in foreground, attempting enhanced recovery...")
                    if not self.enhanced_foreground_recovery():
                        logger.error("Failed to recover app to foreground")
                        continue  # Skip this action if recovery fails
                
                # Perform random action with enhanced monitoring
                action_type, description, element_id, screenshot_paths = self.perform_random_action_with_foreground_management()
                
                # Process all captured screenshots
                if screenshot_paths:
                    logger.info(f"Processing {len(screenshot_paths)} screenshots for action {action_num}...")
                    
                    for i, screenshot_path in enumerate(screenshot_paths):
                        # Analyze screenshot with VisionAI
                        logger.info(f"Running VisionAI analysis for screenshot {i+1}/{len(screenshot_paths)}...")
                        elements = self.analyze_screenshot_with_foreground_management(screenshot_path)
                        
                        # Create app state with step information
                        step_suffix = f"_{i+1}" if len(screenshot_paths) > 1 else ""
                        state_id = f"state_{action_num}{step_suffix}"
                        state = self.create_app_state(state_id, screenshot_path, elements)
                        self.app_states[state_id] = state
                        
                        # Create LLM event for each screenshot
                        event = self.create_llm_event(
                            action_type, description, element_id, screenshot_path
                        )
                        self.llm_events.append(event)
                        
                        # Update current state
                        self.current_state_id = state_id
                        
                        logger.info(f"  ✓ {description} (step {i+1}) -> {len(elements)} elements detected")
                    
                    # Update action count
                    if screenshot_paths:
                        last_screenshot = screenshot_paths[-1]
                        try:
                            new_hash = self._compute_screenshot_hash(last_screenshot)
                        except Exception as hash_err:
                            logger.debug(f"Could not hash screenshot: {hash_err}")
                            new_hash = None

                        if new_hash and (self._last_screenshot_hash is None or new_hash != self._last_screenshot_hash):
                            # State has changed → count this action
                            self.action_count += 1
                            self._last_screenshot_hash = new_hash
                        else:
                            logger.info("State unchanged after action; not counting towards total actions")
                else:
                    logger.warning(f"No screenshots captured for action {action_num}")
                
                # Small delay between actions
                time.sleep(0.2)  # Reduced from 1s - just enough for UI to settle
            
            # Step 6: Save analysis data / reports (optional)
            if self.generate_reports:
                self.save_analysis_data()
            
            # Step 7: Cleanup and uninstall the app after testing
            logger.info("Cleaning up after testing...")
            self.cleanup_after_testing()
            
            logger.info("Uninstalling app...")
            self.uninstall_app()
            
            # Record test end time
            self.test_end_time = datetime.now()
            logger.info(f"Test ended at: {self.test_end_time}")
            
            # Mark testing as completed
            self.testing_completed = True
            
            logger.info("=" * 70)
            logger.info("TESTING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Actions causing state change: {self.changed_action_count}")
            if self.generate_reports:
                logger.info(f"States captured: {len(self.app_states)}")
                logger.info(f"LLM events recorded: {len(self.llm_events)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Testing pipeline failed: {e}")
            # Record test end time even if testing failed
            self.test_end_time = datetime.now()
            logger.info(f"Test ended at: {self.test_end_time} (after failure)")
            return False
    
    def save_analysis_data(self) -> None:
        """Save analysis data for final report generation."""
        logger.info("Saving analysis data...")
        
        analysis_data = {
            "metadata": {
                "app_name": self.app_name,
                "package_name": self.package_name,
                "apk_path": str(self.apk_path),
                "num_actions": self.num_actions,
                "timestamp": datetime.now().isoformat(),
                "total_states": len(self.app_states),
                "total_events": len(self.llm_events)
            },
            "vision_analysis": {
                "screenshots": [
                    {
                        "filename": os.path.basename(state.screenshot_path),
                        "path": state.screenshot_path,
                        "elements_count": len(state.elements),
                        "elements": state.elements
                    }
                    for state in self.app_states.values()
                ]
            },
            "llm_events": [
                {
                    "timestamp": event.timestamp,
                    "state_id": event.state_id,
                    "element_id": event.element_id,
                    "action": event.action,
                    "reasoning": event.reasoning,
                    "resulting_state_id": event.resulting_state_id,
                    "screenshot_path": event.screenshot_path
                }
                for event in self.llm_events
            ]
        }
        
        # Save to file
        data_path = self.output_dir / "analysis_data.json"
        with open(data_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"✓ Analysis data saved: {data_path}")
    
    def generate_final_report(self) -> str:
        """Generate the final comprehensive report.
        
        Returns
        -------
        str
            Path to the generated report

        """
        logger.info("Generating final comprehensive report...")
        
        # Initialize final report generator
        generator = FinalReportGenerator(
            str(self.output_dir / "final_report"),
            app_name=self.app_name,
            package_name=self.package_name
        )
        
        # Set test start and end times if available
        if hasattr(self, 'test_start_time') and self.test_start_time:
            generator.set_test_start_time(self.test_start_time)
        if hasattr(self, 'test_end_time') and self.test_end_time:
            generator.set_test_end_time(self.test_end_time)
        
        # Add LLM events
        for event in self.llm_events:
            generator.add_llm_event(event)
        
        # Add app states
        for state in self.app_states.values():
            generator.add_app_state(state)
        
        # Skip if report generation disabled
        if not self.generate_reports:
            return ""

        # Generate the comprehensive report
        report_path = generator.generate_comprehensive_report()
        
        return report_path

    # Utility: Detect if keyboard is visible (text input active)
    def is_keyboard_visible(self):
        result = subprocess.run(["adb", "shell", "dumpsys", "input_method"], capture_output=True, text=True, check=False)
        return "mInputShown=true" in result.stdout or "mIsInputViewShown=true" in result.stdout

    # Utility: Find and select an empty text field before sending clipboard text
    def select_empty_text_field(self):
        import re
        import xml.etree.ElementTree as ET
        subprocess.run(["adb", "shell", "uiautomator", "dump"], capture_output=True, check=False)
        xml = subprocess.run(["adb", "shell", "cat", "/sdcard/window_dump.xml"], capture_output=True, text=True, check=False).stdout
        try:
            root = ET.fromstring(xml)
            for node in root.iter("node"):
                if node.attrib.get("class") == "android.widget.EditText" and not node.attrib.get("text"):
                    # Tap the center of the empty EditText
                    bounds = node.attrib["bounds"]
                    m = re.match(r"\[(\d+),(\d+)]\[(\d+),(\d+)]", bounds)
                    if m:
                        x = (int(m.group(1)) + int(m.group(3))) // 2
                        y = (int(m.group(2)) + int(m.group(4))) // 2
                        self._tap(x, y)
                        return True
        except Exception as e:
            logger.warning(f"Error selecting empty text field: {e}")
        return False

    # Example usage before sending clipboard text:
    # if not self.is_keyboard_visible():
    #     logger.info("Keyboard not visible, selecting empty text field before input.")
    #     if not self.select_empty_text_field():
    #         logger.warning("No empty text field found, consider tapping a known input element.")
    # else:
    #     logger.info("Keyboard is visible, ready for text input.")

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _get_screen_size(self) -> Tuple[int, int]:
        """Return (width, height) of the connected device. Defaults to 1080x1920."""
        try:
            result = subprocess.run(
                ["adb", "shell", "wm", "size"], capture_output=True, text=True, check=False
            )
            m = re.search(r"(\d+)x(\d+)", result.stdout)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
        # Fallback to common portrait resolution
        return 1080, 1920

    def _scale(self, x: int, y: int) -> Tuple[int, int]:
        """Scale hard-coded 1080x1920 coordinates to current device size."""
        sx = int(x * self.screen_width / 1080)
        sy = int(y * self.screen_height / 1920)
        return sx, sy

    def _tap(self, x: int, y: int) -> None:
        """Convenience wrapper around ADB tap with scaled coordinates."""
        sx, sy = self._scale(x, y)
        subprocess.run(["adb", "shell", "input", "tap", str(sx), str(sy)], capture_output=True, check=False)

    # ------------------------------
    # Text-field autofill helpers
    # ------------------------------

    def _input_text(self, text: str) -> None:
        """Type text into the focused field via adb; handles spaces."""
        escaped = text.replace(" ", "%s")  # adb input text uses %s for space
        subprocess.run(["adb", "shell", "input", "text", escaped], capture_output=True, check=False)

    def _maybe_fill_text_field(self) -> None:
        """If keyboard is visible, insert mock data in round-robin order."""
        try:
            if not self.is_keyboard_visible():
                return
            order = ["name", "mobile", "location", "password"]
            key = order[self._text_fill_index % len(order)]
            value = self.mock_data.get(key, "test")
            logger.info(f"Autofilling text field with mock {key}: {value}")
            self._input_text(value)
            self._text_fill_index += 1
        except Exception as e:
            logger.debug(f"Autofill failed: {e}")

    # ---------------------------------------------------------------------
    # Precision-tap helper
    # ---------------------------------------------------------------------

    def _get_precision_tap_coordinates(self) -> Tuple[int | None, int | None]:
        """Return center (x, y) of a high-priority element from the latest screenshot.

        If VisionEngine or a screenshot is unavailable, returns (None, None).
        """
        if not self.vision_engine:
            return (None, None)

        # Capture a quick screenshot to analyze
        screenshot_path = self.capture_screenshot_optimized("precision_probe", "after", immediate=True)
        if not screenshot_path:
            return (None, None)

        elements = self.vision_engine.analyze(screenshot_path)
        if not elements:
            return (None, None)

        # Filter out social-login buttons
        social_keywords = [
            "google", "facebook", "apple", "twitter", "linkedin", "microsoft"
        ]
        from src.core.element_tracker import get_element_tracker
        tracker = get_element_tracker()
        elements = [
            e for e in elements if not any(sk in e.text.lower() for sk in social_keywords)
            and not tracker.is_element_explored(e)
        ]
        if not elements:
            return (None, None)

        # Simple priority: prefer button-like elements with highest confidence
        elements = sorted(
            elements,
            key=lambda e: (
                0 if e.element_type in ["button", "colored_button", "template"] else 1,
                -e.confidence,
            ),
        )

        top_el = elements[0]
        bbox = top_el.bbox.as_tuple()
        x_c = (bbox[0] + bbox[2]) // 2
        y_c = (bbox[1] + bbox[3]) // 2
        return (x_c, y_c)

    # ------------------------------------------------------------------
    # State-change utilities
    # ------------------------------------------------------------------

    def _compute_screenshot_hash(self, path: str) -> str:
        """Return an MD5 hash of the screenshot contents.

        A cheap way to detect whether the UI truly changed between actions.
        If hashing fails, an empty string is returned so callers can handle it.
        """
        try:
            with open(path, "rb") as fp:
                return hashlib.md5(fp.read()).hexdigest()
        except Exception:
            return ""


def main():
    """Main function to run universal APK testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal APK Tester with DroidBot-GPT + VisionAI")
    parser.add_argument("apk_path", help="Path to the APK file")
    parser.add_argument("--actions", "-a", type=int, default=10,
                       help="Number of actions to perform (default: 10)")
    parser.add_argument("--output-dir", "-o", default="universal_reports",
                       help="Output directory for reports")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip analysis JSON and HTML/PDF report generation for faster runs")
    
    args = parser.parse_args()
    
    # Check if APK exists
    if not Path(args.apk_path).exists():
        logger.error(f"APK not found: {args.apk_path}")
        sys.exit(1)
    
    # Initialize tester
    tester = UniversalAPKTester(
        args.apk_path,
        args.actions,
        args.output_dir,
        generate_reports=not args.no_report
    )
    
    # Run testing pipeline
    success = tester.run_testing_pipeline()
    
    if success:
        if tester.generate_reports:
            report_path = tester.generate_final_report()
            logger.info("🎉 Universal APK testing completed successfully!")
            logger.info(f"📊 Final report available at: {report_path}")
            logger.info(f"📋 Analysis data: {tester.output_dir}/analysis_data.json")
        else:
            logger.info("🎉 Universal APK testing completed successfully (report generation disabled)!")
        logger.info(f"📸 Screenshots: {tester.screenshots_dir}")
        logger.info("🧹 App has been uninstalled and cleaned up")
        logger.info("🔍 VisionAI analysis completed with app foreground management")
    else:
        logger.error("❌ Universal APK testing failed!")
        # Attempt cleanup even if testing failed
        try:
            tester.cleanup_after_testing()
            tester.uninstall_app()
            logger.info("🧹 Cleanup performed despite test failure")
        except Exception as e:
            logger.warning(f"Cleanup failed after test failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 