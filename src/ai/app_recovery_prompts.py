"""Intuitive App Recovery Prompts for Explorer framework.

This module provides intelligent prompts for app recovery scenarios, helping
the system understand when and how to recover apps that leave the foreground.
The prompts are designed to be context-aware and provide clear guidance for
different recovery situations.
"""

from typing import Dict, List, Any, Optional
from enum import Enum


class RecoveryScenario(Enum):
    """Different recovery scenarios that require different approaches."""
    APP_BACKGROUNDED = "app_backgrounded"
    APP_CRASHED = "app_crashed"
    APP_FORCE_CLOSED = "app_force_closed"
    SYSTEM_DIALOG = "system_dialog"
    EXTERNAL_APP_LAUNCHED = "external_app_launched"
    PERMISSION_DENIED = "permission_denied"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class AppRecoveryPrompts:
    """Intuitive prompt system for app recovery scenarios."""
    
    def __init__(self):
        """Initialize the app recovery prompts."""
        self.base_system_prompt = self._get_base_system_prompt()
        self.scenario_prompts = self._get_scenario_prompts()
        self.recovery_strategies = self._get_recovery_strategies()
    
    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt for app recovery."""
        return (
            "You are an expert Android app recovery specialist with deep knowledge of "
            "mobile app lifecycle management, foreground/background transitions, and "
            "system-level app control mechanisms.\n\n"
            
            "Your role is to:\n"
            "1. ANALYZE app state changes and determine recovery strategies\n"
            "2. PROVIDE intelligent guidance for app restoration\n"
            "3. UNDERSTAND when recovery is necessary vs. when to let the app stay backgrounded\n"
            "4. RECOMMEND appropriate recovery methods based on the scenario\n"
            "5. CONSIDER user experience and app stability in recovery decisions\n\n"
            
            "CRITICAL RECOVERY PRINCIPLES:\n"
            "• ALWAYS prioritize user experience - don't interrupt user actions unnecessarily\n"
            "• UNDERSTAND the difference between temporary backgrounding and permanent app loss\n"
            "• USE the least intrusive recovery method that will be effective\n"
            "• CONSIDER app state and automation context when making recovery decisions\n"
            "• MONITOR recovery success and adapt strategies accordingly\n"
            "• AVOID aggressive recovery that might interfere with legitimate user actions\n\n"
            
            "RECOVERY DECISION FRAMEWORK:\n"
            "1. ASSESS the current app state and context\n"
            "2. IDENTIFY the likely cause of app leaving foreground\n"
            "3. DETERMINE if recovery is necessary and appropriate\n"
            "4. SELECT the most suitable recovery strategy\n"
            "5. EXECUTE recovery with appropriate timing and method\n"
            "6. VERIFY recovery success and adjust if needed\n\n"
            
            "OUTPUT FORMAT:\n"
            "Provide JSON response with:\n"
            "- recovery_needed: boolean indicating if recovery is required\n"
            "- scenario: the identified recovery scenario\n"
            "- recommended_strategy: the best recovery approach\n"
            "- reasoning: explanation of the decision\n"
            "- confidence: confidence level in the recommendation (0.0-1.0)\n"
            "- additional_actions: any additional steps needed\n"
        )
    
    def _get_scenario_prompts(self) -> Dict[RecoveryScenario, str]:
        """Get specific prompts for different recovery scenarios."""
        return {
            RecoveryScenario.APP_BACKGROUNDED: (
                "APP BACKGROUNDED SCENARIO:\n"
                "The target app has moved to the background but is still running.\n"
                "This is often normal behavior and may not require immediate recovery.\n\n"
                "CONSIDERATIONS:\n"
                "• Is this a temporary backgrounding (user checking notifications, etc.)?\n"
                "• Is the automation task still in progress?\n"
                "• Would recovery interrupt legitimate user actions?\n"
                "• How long has the app been backgrounded?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• If automation is active and app is needed: Use gentle recovery (foreground service)\n"
                "• If user might be interacting: Wait and monitor\n"
                "• If backgrounding is prolonged: Consider app launch recovery\n"
            ),
            
            RecoveryScenario.APP_CRASHED: (
                "APP CRASHED SCENARIO:\n"
                "The target app has crashed or stopped unexpectedly.\n"
                "This requires immediate recovery to continue automation.\n\n"
                "CONSIDERATIONS:\n"
                "• What caused the crash? (memory, network, system issue)\n"
                "• Is the app still installed and accessible?\n"
                "• Will a simple restart resolve the issue?\n"
                "• Are there any error messages or logs available?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• Use force stop and restart strategy\n"
                "• Clear app cache if necessary\n"
                "• Check for system-level issues\n"
                "• Verify app installation status\n"
            ),
            
            RecoveryScenario.APP_FORCE_CLOSED: (
                "APP FORCE CLOSED SCENARIO:\n"
                "The target app has been force closed by the user or system.\n"
                "This requires careful recovery to avoid user frustration.\n\n"
                "CONSIDERATIONS:\n"
                "• Was this intentional user action?\n"
                "• Is automation still needed?\n"
                "• Should we respect the user's decision?\n"
                "• How critical is the automation task?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• If automation is critical: Use app launch recovery\n"
                "• If user likely closed intentionally: Pause automation\n"
                "• Use gentle recovery methods to avoid interference\n"
                "• Consider user notification about automation status\n"
            ),
            
            RecoveryScenario.SYSTEM_DIALOG: (
                "SYSTEM DIALOG SCENARIO:\n"
                "A system dialog (permissions, updates, etc.) has appeared.\n"
                "The app may be temporarily backgrounded due to system overlay.\n\n"
                "CONSIDERATIONS:\n"
                "• What type of system dialog is present?\n"
                "• Can the dialog be dismissed or handled?\n"
                "• Is user interaction required?\n"
                "• Will the app return automatically?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• Wait for dialog to be resolved naturally\n"
                "• If dialog persists: Use gentle recovery\n"
                "• Handle permission requests if possible\n"
                "• Avoid interrupting system processes\n"
            ),
            
            RecoveryScenario.EXTERNAL_APP_LAUNCHED: (
                "EXTERNAL APP LAUNCHED SCENARIO:\n"
                "An external app (camera, gallery, browser, etc.) has been launched.\n"
                "The target app is backgrounded but may return automatically.\n\n"
                "CONSIDERATIONS:\n"
                "• What external app was launched?\n"
                "• Is this part of the automation flow?\n"
                "• Will the user return to the target app?\n"
                "• How long should we wait before recovery?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• Wait for natural return to target app\n"
                "• If prolonged: Use recent apps recovery\n"
                "• Consider if external app interaction is needed\n"
                "• Use gentle recovery methods\n"
            ),
            
            RecoveryScenario.PERMISSION_DENIED: (
                "PERMISSION DENIED SCENARIO:\n"
                "App permissions have been denied, causing app to background or crash.\n"
                "This requires permission handling before recovery.\n\n"
                "CONSIDERATIONS:\n"
                "• Which permissions were denied?\n"
                "• Can permissions be granted programmatically?\n"
                "• Is the app functional without these permissions?\n"
                "• Should automation continue or pause?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• Handle permission requests if possible\n"
                "• Use app settings navigation if needed\n"
                "• Restart app after permission changes\n"
                "• Consider alternative approaches without permissions\n"
            ),
            
            RecoveryScenario.NETWORK_ERROR: (
                "NETWORK ERROR SCENARIO:\n"
                "Network connectivity issues have caused app problems.\n"
                "App may be backgrounded or showing error states.\n\n"
                "CONSIDERATIONS:\n"
                "• Is network connectivity available?\n"
                "• Are there specific network errors?\n"
                "• Can the app function offline?\n"
                "• Should we wait for network recovery?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• Wait for network recovery if possible\n"
                "• Use app restart if network-dependent features fail\n"
                "• Consider offline mode if available\n"
                "• Monitor network status and retry\n"
            ),
            
            RecoveryScenario.UNKNOWN_ERROR: (
                "UNKNOWN ERROR SCENARIO:\n"
                "App state is unclear or unexpected behavior has occurred.\n"
                "Requires careful analysis and conservative recovery approach.\n\n"
                "CONSIDERATIONS:\n"
                "• What information is available about the error?\n"
                "• Is the app still installed and accessible?\n"
                "• Are there any error logs or messages?\n"
                "• Should we use conservative or aggressive recovery?\n\n"
                "RECOVERY GUIDANCE:\n"
                "• Start with gentle recovery methods\n"
                "• Escalate to more aggressive methods if needed\n"
                "• Monitor recovery success carefully\n"
                "• Consider system-level issues\n"
            )
        }
    
    def _get_recovery_strategies(self) -> Dict[str, str]:
        """Get descriptions of different recovery strategies."""
        return {
            "foreground_service": (
                "Use foreground service to bring app back to foreground. "
                "This is the gentlest method and works well for apps that "
                "support foreground services."
            ),
            "app_launch": (
                "Launch the app directly using package manager. "
                "This is effective for apps that are installed but not running."
            ),
            "recent_apps": (
                "Use recent apps switcher to find and restore the app. "
                "This works well for apps that are still in recent apps list."
            ),
            "home_and_launch": (
                "Go to home screen and then launch the app. "
                "This clears any overlays and provides a clean launch."
            ),
            "force_stop_restart": (
                "Force stop the app and then restart it. "
                "This is more aggressive but effective for crashed apps."
            ),
            "clear_recents_launch": (
                "Clear recent apps and then launch the target app. "
                "This is the most aggressive method and should be used sparingly."
            ),
            "wait_and_monitor": (
                "Wait for the app to return naturally and monitor the situation. "
                "This is appropriate when app loss is likely temporary."
            ),
            "pause_automation": (
                "Pause automation and wait for user intervention. "
                "This is appropriate when recovery might interfere with user actions."
            )
        }
    
    def build_recovery_analysis_prompt(
        self,
        target_package: str,
        current_app_state: str,
        previous_app_state: str,
        automation_context: str,
        recovery_history: List[Dict[str, Any]],
        system_info: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Build a prompt for analyzing recovery needs.
        
        Args:
            target_package: Package name of the target app
            current_app_state: Current state of the app
            previous_app_state: Previous state of the app
            automation_context: Current automation task context
            recovery_history: History of previous recovery attempts
            system_info: System information (network, battery, etc.)
            
        Returns:
            List of message dictionaries for OpenAI API
        """
        # Determine the likely scenario
        scenario = self._identify_scenario(current_app_state, previous_app_state, system_info)
        scenario_prompt = self.scenario_prompts.get(scenario, self.scenario_prompts[RecoveryScenario.UNKNOWN_ERROR])
        
        # Build the user message
        user_content = (
            f"TARGET APP: {target_package}\n"
            f"STATE CHANGE: {previous_app_state} → {current_app_state}\n"
            f"AUTOMATION CONTEXT: {automation_context}\n\n"
            
            f"RECOVERY HISTORY ({len(recovery_history)} attempts):\n"
        )
        
        if recovery_history:
            for i, attempt in enumerate(recovery_history[-3:], 1):  # Last 3 attempts
                user_content += (
                    f"{i}. Strategy: {attempt.get('strategy', 'Unknown')} | "
                    f"Success: {attempt.get('success', False)} | "
                    f"Duration: {attempt.get('duration', 0):.2f}s\n"
                )
        else:
            user_content += "No previous recovery attempts\n"
        
        user_content += f"\nSYSTEM INFO:\n"
        for key, value in system_info.items():
            user_content += f"• {key}: {value}\n"
        
        user_content += (
            f"\nSCENARIO ANALYSIS:\n{scenario_prompt}\n\n"
            
            "ANALYSIS REQUEST:\n"
            "Based on the current situation, determine:\n"
            "1. Is recovery needed at this time?\n"
            "2. What is the most likely scenario?\n"
            "3. Which recovery strategy would be most appropriate?\n"
            "4. What is the confidence level in this recommendation?\n"
            "5. Are there any additional actions needed?\n\n"
            
            "Provide your analysis in JSON format with recovery_needed, scenario, "
            "recommended_strategy, reasoning, confidence, and additional_actions."
        )
        
        return [
            {"role": "system", "content": self.base_system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def _identify_scenario(
        self,
        current_state: str,
        previous_state: str,
        system_info: Dict[str, Any]
    ) -> RecoveryScenario:
        """Identify the most likely recovery scenario based on state changes and system info.
        
        Args:
            current_state: Current app state
            previous_state: Previous app state
            system_info: System information
            
        Returns:
            RecoveryScenario: The identified scenario
        """
        # Check for specific indicators in system info
        if system_info.get('network_status') == 'disconnected':
            return RecoveryScenario.NETWORK_ERROR
        
        if system_info.get('permission_denied'):
            return RecoveryScenario.PERMISSION_DENIED
        
        if system_info.get('system_dialog_active'):
            return RecoveryScenario.SYSTEM_DIALOG
        
        if system_info.get('external_app_launched'):
            return RecoveryScenario.EXTERNAL_APP_LAUNCHED
        
        # Analyze state transitions
        if previous_state == 'foreground' and current_state == 'background':
            return RecoveryScenario.APP_BACKGROUNDED
        
        if current_state in ['stopped', 'crashed']:
            return RecoveryScenario.APP_CRASHED
        
        if current_state == 'force_closed':
            return RecoveryScenario.APP_FORCE_CLOSED
        
        # Default to unknown if we can't determine
        return RecoveryScenario.UNKNOWN_ERROR
    
    def build_recovery_execution_prompt(
        self,
        target_package: str,
        scenario: RecoveryScenario,
        strategy: str,
        previous_attempts: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Build a prompt for executing recovery strategies.
        
        Args:
            target_package: Package name of the target app
            scenario: The identified recovery scenario
            strategy: The chosen recovery strategy
            previous_attempts: Previous recovery attempts
            
        Returns:
            List of message dictionaries for OpenAI API
        """
        strategy_description = self.recovery_strategies.get(strategy, "Unknown strategy")
        
        user_content = (
            f"RECOVERY EXECUTION REQUEST:\n"
            f"Target App: {target_package}\n"
            f"Scenario: {scenario.value}\n"
            f"Strategy: {strategy}\n"
            f"Strategy Description: {strategy_description}\n\n"
            
            f"PREVIOUS ATTEMPTS ({len(previous_attempts)}):\n"
        )
        
        if previous_attempts:
            for i, attempt in enumerate(previous_attempts[-3:], 1):
                user_content += (
                    f"{i}. Strategy: {attempt.get('strategy', 'Unknown')} | "
                    f"Success: {attempt.get('success', False)} | "
                    f"Error: {attempt.get('error_message', 'None')}\n"
                )
        else:
            user_content += "No previous attempts\n"
        
        user_content += (
            f"\nEXECUTION GUIDANCE:\n"
            f"• Follow the strategy description carefully\n"
            f"• Monitor for success indicators\n"
            f"• Be prepared to try alternative strategies if this fails\n"
            f"• Consider timing and user experience\n"
            f"• Log all actions and results\n\n"
            
            "Provide execution plan with:\n"
            "- steps: List of specific steps to execute\n"
            "- success_indicators: How to verify recovery success\n"
            "- fallback_strategy: Alternative if this fails\n"
            "- timing: When to execute each step\n"
            "- monitoring: What to watch for during execution"
        )
        
        return [
            {"role": "system", "content": self.base_system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def get_recovery_guidance(self, scenario: RecoveryScenario) -> Dict[str, Any]:
        """Get guidance for a specific recovery scenario.
        
        Args:
            scenario: The recovery scenario
            
        Returns:
            Dictionary with guidance information
        """
        return {
            "scenario": scenario.value,
            "description": self.scenario_prompts.get(scenario, ""),
            "recommended_strategies": self._get_recommended_strategies(scenario),
            "considerations": self._get_scenario_considerations(scenario),
            "success_indicators": self._get_success_indicators(scenario)
        }
    
    def _get_recommended_strategies(self, scenario: RecoveryScenario) -> List[str]:
        """Get recommended strategies for a scenario."""
        strategy_map = {
            RecoveryScenario.APP_BACKGROUNDED: ["foreground_service", "app_launch", "wait_and_monitor"],
            RecoveryScenario.APP_CRASHED: ["force_stop_restart", "app_launch"],
            RecoveryScenario.APP_FORCE_CLOSED: ["app_launch", "recent_apps", "pause_automation"],
            RecoveryScenario.SYSTEM_DIALOG: ["wait_and_monitor", "foreground_service"],
            RecoveryScenario.EXTERNAL_APP_LAUNCHED: ["wait_and_monitor", "recent_apps"],
            RecoveryScenario.PERMISSION_DENIED: ["app_launch", "pause_automation"],
            RecoveryScenario.NETWORK_ERROR: ["wait_and_monitor", "app_launch"],
            RecoveryScenario.UNKNOWN_ERROR: ["foreground_service", "app_launch", "force_stop_restart"]
        }
        return strategy_map.get(scenario, ["app_launch"])
    
    def _get_scenario_considerations(self, scenario: RecoveryScenario) -> List[str]:
        """Get considerations for a scenario."""
        considerations_map = {
            RecoveryScenario.APP_BACKGROUNDED: [
                "Check if automation is still active",
                "Consider if user is interacting with other apps",
                "Monitor for natural return to foreground"
            ],
            RecoveryScenario.APP_CRASHED: [
                "Check for system resources (memory, storage)",
                "Look for error logs or crash reports",
                "Consider app-specific issues"
            ],
            RecoveryScenario.APP_FORCE_CLOSED: [
                "Respect user intent if intentional",
                "Check automation criticality",
                "Use gentle recovery methods"
            ],
            RecoveryScenario.SYSTEM_DIALOG: [
                "Wait for dialog resolution",
                "Avoid interrupting system processes",
                "Handle permissions if needed"
            ],
            RecoveryScenario.EXTERNAL_APP_LAUNCHED: [
                "Check if external app interaction is needed",
                "Wait for natural return",
                "Use recent apps if prolonged"
            ],
            RecoveryScenario.PERMISSION_DENIED: [
                "Handle permission requests",
                "Check app functionality without permissions",
                "Consider user notification"
            ],
            RecoveryScenario.NETWORK_ERROR: [
                "Monitor network status",
                "Check for offline functionality",
                "Wait for network recovery"
            ],
            RecoveryScenario.UNKNOWN_ERROR: [
                "Start with conservative approaches",
                "Monitor system state",
                "Escalate if needed"
            ]
        }
        return considerations_map.get(scenario, ["Monitor situation", "Use conservative approach"])
    
    def _get_success_indicators(self, scenario: RecoveryScenario) -> List[str]:
        """Get success indicators for a scenario."""
        indicators_map = {
            RecoveryScenario.APP_BACKGROUNDED: [
                "App returns to foreground",
                "Automation can continue",
                "No user interruption"
            ],
            RecoveryScenario.APP_CRASHED: [
                "App launches successfully",
                "No crash on startup",
                "Core functionality works"
            ],
            RecoveryScenario.APP_FORCE_CLOSED: [
                "App launches without issues",
                "User experience not disrupted",
                "Automation can resume"
            ],
            RecoveryScenario.SYSTEM_DIALOG: [
                "Dialog is resolved",
                "App returns to foreground",
                "No permission issues"
            ],
            RecoveryScenario.EXTERNAL_APP_LAUNCHED: [
                "Return to target app",
                "External app interaction complete",
                "Automation can continue"
            ],
            RecoveryScenario.PERMISSION_DENIED: [
                "Permissions granted",
                "App functions normally",
                "No permission errors"
            ],
            RecoveryScenario.NETWORK_ERROR: [
                "Network connectivity restored",
                "App functions normally",
                "No network-related errors"
            ],
            RecoveryScenario.UNKNOWN_ERROR: [
                "App state is stable",
                "Automation can continue",
                "No recurring issues"
            ]
        }
        return indicators_map.get(scenario, ["App is accessible", "No errors", "Functionality restored"])


# Global instance
_app_recovery_prompts: Optional[AppRecoveryPrompts] = None


def get_app_recovery_prompts() -> AppRecoveryPrompts:
    """Get the global app recovery prompts instance.
    
    Returns:
        AppRecoveryPrompts: The prompts instance
    """
    global _app_recovery_prompts
    if _app_recovery_prompts is None:
        _app_recovery_prompts = AppRecoveryPrompts()
    return _app_recovery_prompts 