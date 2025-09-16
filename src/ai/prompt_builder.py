"""Enhanced prompt builder that assembles messages for OpenAI chat completion with improved element detection."""

from __future__ import annotations

from typing import Any

from ..core.state_serializer import serialize_state

_SYSTEM_PROMPT = (
    "You are \"AppCritic-GPT\", a senior mobile-app reviewer and UI automation expert "
    "with deep knowledge of Android UI patterns, element detection, and user interaction flows. "
    "You excel at identifying interactive elements, understanding app context, and providing "
    "actionable insights for automation tasks."
)

# Enhanced instructions for better element detection and analysis
_INSTRUCTIONS = (
    "\nInstructions:\n"
    "1. ANALYZE UI ELEMENTS:\n"
    "   - Identify interactive elements (buttons, inputs, links, navigation)\n"
    "   - Classify element types and their intended functions\n"
    "   - Detect text input fields and their expected content types\n"
    "   - Recognize navigation patterns and app structure\n"
    "\n"
    "2. ELEMENT CLASSIFICATION:\n"
    "   - BUTTONS: Action buttons, navigation buttons, toggle buttons\n"
    "   - INPUTS: Text fields, search boxes, form inputs, selection controls\n"
    "   - NAVIGATION: Menu items, tabs, breadcrumbs, back/forward buttons\n"
    "   - CONTENT: Headers, labels, descriptions, status indicators\n"
    "   - SYSTEM: Status bar, navigation bar, system controls\n"
    "\n"
    "3. CONTEXT ANALYSIS:\n"
    "   - Understand the current app screen context\n"
    "   - Identify the user's likely next action\n"
    "   - Recognize form flows and multi-step processes\n"
    "   - Detect error states and validation messages\n"
    "\n"
    "4. INTERACTIVE INTENT:\n"
    "   - Determine which elements are meant to be clicked/tapped\n"
    "   - Identify elements that require text input\n"
    "   - Recognize elements that trigger navigation or actions\n"
    "   - Understand element hierarchy and relationships\n"
    "\n"
    "5. AUTOMATION GUIDANCE:\n"
    "   - Suggest the most logical next action for automation\n"
    "   - Identify potential input values for text fields\n"
    "   - Recognize completion states and success indicators\n"
    "   - Detect error conditions that need handling\n"
    "\n"
    "6. ACTION PRIORITY RULES:\n"
    "   - PRIORITIZE UNEXPLORED ELEMENTS: Always prefer elements that haven't been interacted with\n"
    "   - TEXT INPUT PRIORITY: Input fields should be prioritized over buttons when both are available\n"
    "   - NAVIGATION PRIORITY: Navigation elements (back, menu, settings) should be lower priority than action elements\n"
    "   - FORM COMPLETION: When filling forms, prioritize required fields over optional ones\n"
    "   - ERROR HANDLING: Error messages and validation issues should be addressed before continuing\n"
    "   - PROGRESS INDICATORS: Elements that show progress or completion should be noted but not prioritized for interaction\n"
    "\n"
    "7. FOREGROUND RETENTION STRATEGY:\n"
    "   - PRIORITIZE IN-APP ELEMENTS: Focus on elements within the current app interface\n"
    "   - AVOID PERMANENT FOREGROUND LOSS: Skip elements that might cause the app to close or be uninstalled\n"
    "   - ALLOW TEMPORARY FOREGROUND LOSS: Can interact with elements that temporarily leave the app (camera, gallery, share) if recovery is available\n"
    "   - FOCUS ON CORE FUNCTIONALITY: Prioritize main app features over settings, help, or external integrations\n"
    "   - SMART RISK ASSESSMENT: Evaluate if an action will return to the app or cause permanent loss\n"
    "   - PREFER IN-APP ALTERNATIVES: When possible, choose in-app options over external ones\n"
    "   - CONTEXT-AWARE DECISIONS: Consider the current app state and recovery capabilities\n"
    "\n"
    "8. OUTPUT FORMAT:\n"
    "   - Provide detailed element analysis with confidence levels\n"
    "   - Include element relationships and hierarchy\n"
    "   - Suggest appropriate actions for automation\n"
    "   - Include stuck detection analysis and recovery recommendations\n"
    "   - Conclude with EXACTLY 50 words summary.\n\n"
    "Output JSON keys: element_analysis, interactive_elements, suggested_actions, context_insights, stuck_analysis, summary_50_words."
    "\n"
    "9. ANALYSIS & RECOMMENDATIONS:\n"
    "   - At the end of your JSON output, add a key named 'analysis_recommendations' whose value is **five numbered sections** exactly as follows: \n"
    "     1. UI/UX analysis and observations (2-3 sentences)\n"
    "     2. Potential accessibility improvements (2-3 sentences)\n"
    "     3. Automation opportunities (2-3 sentences)\n"
    "     4. Testing recommendations (2-3 sentences)\n"
    "     5. Overall app quality assessment (2-3 sentences)\n"
    "   - Keep each sentence concise.\n"
)

# Specialized prompts for different analysis types
_ELEMENT_DETECTION_PROMPT = (
    "\nELEMENT DETECTION FOCUS:\n"
    "1. VISUAL PATTERNS:\n"
    "   - Look for button-like shapes (rounded rectangles, borders)\n"
    "   - Identify input field indicators (underlines, placeholders)\n"
    "   - Detect navigation elements (icons, arrows, tabs)\n"
    "   - Recognize interactive colors (blue links, colored buttons)\n"
    "\n"
    "2. TEXT PATTERNS:\n"
    "   - Action words: 'Login', 'Submit', 'Continue', 'Next', 'Save'\n"
    "   - Input hints: 'Enter email', 'Type here', 'Search', 'Password'\n"
    "   - Navigation: 'Back', 'Menu', 'Settings', 'Profile'\n"
    "   - Status: 'Loading', 'Error', 'Success', 'Complete'\n"
    "\n"
    "3. SYMBOL DETECTION:\n"
    "   - Look for meaningful UI symbols: × (close/cancel), ✓ (check/confirm), → (next/forward), ← (back/previous)\n"
    "   - Identify action symbols: + (add), - (remove), ⚙️ (settings), 🔍 (search), ❤️ (favorite/like)\n"
    "   - Detect navigation symbols: ⋮ (menu), ⬅️ (back), ➡️ (forward), ⬆️ (up), ⬇️ (down)\n"
    "   - Recognize status symbols: ⚠️ (warning), ❌ (error), ✅ (success), 🔄 (refresh/loading)\n"
    "   - Look for app-specific symbols: 🛒 (cart), 💰 (payment), 📱 (phone), 📧 (email), 🔐 (security)\n"
    "   - Identify interactive symbols that serve as buttons or links\n"
    "   - Pay attention to symbol placement and context for meaning\n"
    "\n"
    "4. LAYOUT PATTERNS:\n"
    "   - Bottom navigation bars\n"
    "   - Floating action buttons\n"
    "   - Form layouts with labels and inputs\n"
    "   - List items with action buttons\n"
    "   - Modal dialogs and overlays\n"
)

_CONTEXT_ANALYSIS_PROMPT = (
    "\nCONTEXT ANALYSIS:\n"
    "1. SCREEN TYPE IDENTIFICATION:\n"
    "   - Login/Registration screens\n"
    "   - Main app screens (home, dashboard)\n"
    "   - Form screens (settings, profile, checkout)\n"
    "   - List/Detail screens\n"
    "   - Error/Confirmation screens\n"
    "\n"
    "2. USER FLOW UNDERSTANDING:\n"
    "   - Current step in multi-step processes\n"
    "   - Required vs optional actions\n"
    "   - Dependencies between elements\n"
    "   - Completion criteria\n"
    "\n"
    "3. APP-SPECIFIC PATTERNS:\n"
    "   - E-commerce: product listings, cart, checkout\n"
    "   - Social: feed, profile, messaging\n"
    "   - Utility: settings, tools, preferences\n"
    "   - Content: articles, media, search\n"
)

_FOREGROUND_RETENTION_PROMPT = (
    "\nFOREGROUND RETENTION ANALYSIS:\n"
    "1. ELEMENT RISK ASSESSMENT:\n"
    "   - CRITICAL RISK: Elements that might cause permanent app closure or uninstallation\n"
    "   - HIGH RISK: Elements that might trigger system dialogs or external apps with uncertain return\n"
    "   - MEDIUM RISK: Elements that temporarily leave the app but return (camera, gallery, share)\n"
    "   - LOW RISK: Elements that stay within the current app context\n"
    "   - SAFE: Core app functionality elements (buttons, inputs, navigation within app)\n"
    "\n"
    "2. ELEMENTS TO AVOID (PERMANENT FOREGROUND LOSS):\n"
    "   - 'Uninstall' or 'Remove app' buttons\n"
    "   - 'Clear data' or 'Reset app' buttons\n"
    "   - 'Force stop' or 'Kill app' buttons\n"
    "   - Elements that might crash or close the app permanently\n"
    "   - Deep links to other apps that don't return\n"
    "\n"
    "3. ELEMENTS THAT CAN BE INTERACTED WITH (TEMPORARY LOSS):\n"
    "   - 'Share' buttons or icons (returns to app)\n"
    "   - 'Camera' or 'Gallery' buttons (returns to app)\n"
    "   - 'Location' or 'Map' buttons (usually returns)\n"
    "   - 'Settings' or 'Preferences' buttons (returns to app)\n"
    "   - 'Help' or 'Support' buttons (returns to app)\n"
    "   - 'Rate app' or 'Feedback' buttons (returns to app)\n"
    "   - 'Export' or 'Save to' buttons (returns to app)\n"
    "   - 'Call' or 'Email' buttons (returns to app)\n"
    "   - 'Social media' integration buttons (returns to app)\n"
    "   - External links that open and return\n"
    "\n"
    "4. PRIORITIZE THESE ELEMENT TYPES:\n"
    "   - Form input fields (email, password, text)\n"
    "   - Action buttons (Submit, Continue, Next, Save)\n"
    "   - Navigation within app (tabs, menu items)\n"
    "   - List items and content elements\n"
    "   - Toggle switches and checkboxes\n"
    "   - Search functionality within app\n"
    "   - Product/service selection elements\n"
    "\n"
    "5. CONTEXT-AWARE DECISIONS:\n"
    "   - If on a form screen: Focus on completing the form\n"
    "   - If on a product screen: Focus on product interactions\n"
    "   - If on a list screen: Focus on list item selection\n"
    "   - If on a detail screen: Focus on detail view interactions\n"
    "   - Prefer in-app actions but allow temporary external interactions\n"
    "   - Consider app recovery capabilities when making decisions\n"
)

_KEYBOARD_INTERACTION_PROMPT = (
    "\nKEYBOARD DETECTION AND INTERACTION:\n"
    "1. KEYBOARD IDENTIFICATION:\n"
    "   - Look for on-screen keyboard layouts at the bottom of the screen\n"
    "   - Identify keyboard rows: numbers (1-9,0), letters (QWERTY layout), symbols, space bar\n"
    "   - Detect special keys: Enter, Backspace, Shift, Caps Lock, Numbers/Symbols toggle\n"
    "   - Recognize keyboard themes: light, dark, or custom app themes\n"
    "   - Look for keyboard indicators: cursor position, text selection, auto-complete suggestions\n"
    "\n"
    "2. KEYBOARD LAYOUT PATTERNS:\n"
    "   - QWERTY layout: Standard letter arrangement\n"
    "   - Number row: 1-9,0 at the top\n"
    "   - Symbol row: Common punctuation and symbols\n"
    "   - Space bar: Usually the largest key at the bottom\n"
    "   - Function keys: Enter, Backspace, Shift, etc.\n"
    "   - Language indicators: If multiple languages are available\n"
    "\n"
    "3. TEXT INPUT STRATEGY:\n"
    "   - PRIORITIZE text input fields when keyboard is visible\n"
    "   - TAP on input field to focus and show keyboard\n"
    "   - USE keyboard to type text character by character\n"
    "   - TAP individual keys in sequence to form words\n"
    "   - USE space bar between words\n"
    "   - USE backspace to correct mistakes\n"
    "   - TAP Enter/Submit when done typing\n"
    "\n"
    "4. KEYBOARD INTERACTION RULES:\n"
    "   - TAP keys in reading order (left to right, top to bottom)\n"
    "   - WAIT briefly between key taps for responsiveness\n"
    "   - USE Shift key for capital letters when needed\n"
    "   - SWITCH to numbers/symbols for special characters\n"
    "   - TAP Enter to submit or move to next field\n"
    "   - TAP Backspace to delete characters\n"
    "   - TAP outside keyboard to dismiss if needed\n"
    "\n"
    "5. COMMON TEXT INPUT SCENARIOS:\n"
    "   - EMAIL: Type email address with @ symbol\n"
    "   - PASSWORD: Type password characters (may be hidden)\n"
    "   - NAME: Type first and last name with space\n"
    "   - PHONE: Type numbers only\n"
    "   - ADDRESS: Type full address with spaces and punctuation\n"
    "   - SEARCH: Type search terms and tap search button\n"
    "   - MESSAGE: Type longer text with multiple words\n"
    "\n"
    "6. KEYBOARD NAVIGATION:\n"
    "   - TAP input field to show keyboard\n"
    "   - TAP 'Next' or 'Continue' to move between fields\n"
    "   - TAP 'Done' or 'Submit' to complete form\n"
    "   - TAP 'Cancel' or 'Back' to exit without saving\n"
    "   - TAP outside keyboard area to hide keyboard\n"
    "\n"
    "7. SMART TEXT GENERATION:\n"
    "   - Generate realistic email addresses (user@domain.com)\n"
    "   - Create appropriate passwords (mix of letters, numbers, symbols)\n"
    "   - Use realistic names (First Last format)\n"
    "   - Generate valid phone numbers (10-11 digits)\n"
    "   - Create believable addresses with street, city, state, zip\n"
    "   - Write relevant search terms for the app context\n"
    "   - Compose meaningful messages or comments\n"
)

_STUCK_DETECTION_PROMPT = (
    "\nSTUCK DETECTION AND RECOVERY:\n"
    "1. VISION ANALYSIS STUCK INDICATORS:\n"
    "   - REPEATED IDENTICAL SCREENS: Same UI elements appearing multiple times without progress\n"
    "   - NO INTERACTIVE ELEMENTS: Screen shows only static content with no clickable elements\n"
    "   - LOADING STATES: Persistent loading indicators, spinners, or progress bars that don't complete\n"
    "   - ERROR MESSAGES: Repeated error dialogs, network errors, or validation failures\n"
    "   - BLANK SCREENS: Empty or white screens with no visible content\n"
    "   - CRASH INDICATORS: App appears frozen, unresponsive, or shows crash dialogs\n"
    "   - PERMISSION LOOPS: Repeated permission requests that can't be satisfied\n"
    "   - SYSTEM DIALOGS: Persistent system alerts that block app interaction\n"
    "\n"
    "2. ACTION HISTORY ANALYSIS:\n"
    "   - REPEATED ACTIONS: Same action performed multiple times without effect\n"
    "   - NO PROGRESS: Actions that don't advance the app state or complete tasks\n"
    "   - CIRCULAR NAVIGATION: Moving between same screens repeatedly\n"
    "   - FAILED INTERACTIONS: Actions that result in errors or no response\n"
    "   - TIMEOUT PATTERNS: Actions that take too long or never complete\n"
    "   - INVALID INPUT: Text input that causes validation errors or crashes\n"
    "   - MISSING ELEMENTS: Expected elements not found after multiple attempts\n"
    "\n"
    "3. COMBINED STUCK SCENARIOS:\n"
    "   - VISION + ACTION MISMATCH: Elements detected but actions fail repeatedly\n"
    "   - STATE INCONSISTENCY: Vision shows one state but actions reflect another\n"
    "   - ELEMENT DETECTION FAILURE: Critical elements not detected despite being visible\n"
    "   - ACTION EXECUTION FAILURE: Detected elements can't be interacted with\n"
    "   - TIMING ISSUES: Elements appear/disappear too quickly for reliable interaction\n"
    "   - COORDINATE MISALIGNMENT: Taps not hitting intended elements\n"
    "   - FOREGROUND CONFLICTS: App switching between foreground/background during analysis\n"
    "\n"
    "4. STUCK RECOVERY STRATEGIES:\n"
    "   - ALTERNATIVE ELEMENTS: Try different interactive elements on the same screen\n"
    "   - NAVIGATION RESET: Use back button, home button, or app restart to reset state\n"
    "   - INPUT VARIATION: Try different text inputs, formats, or values\n"
    "   - TIMING ADJUSTMENT: Wait longer between actions or try faster sequences\n"
    "   - ELEMENT PRIORITY SHIFT: Focus on different element types (buttons vs inputs)\n"
    "   - CONTEXT SWITCH: Move to different app sections or screens\n"
    "   - FORCE REFRESH: Pull-to-refresh, swipe gestures, or manual screen refresh\n"
    "   - SYSTEM RECOVERY: Use device back button, recent apps, or app switcher\n"
    "\n"
    "5. STUCK PREVENTION:\n"
    "   - DIVERSIFY ACTIONS: Avoid repeating the same action multiple times\n"
    "   - VALIDATE PROGRESS: Check if actions actually advance the app state\n"
    "   - MONITOR RESPONSES: Pay attention to app reactions and error messages\n"
    "   - ADAPT TO CHANGES: Modify strategy when app behavior changes\n"
    "   - USE FALLBACKS: Have alternative approaches for common failure points\n"
    "   - TIMEOUT HANDLING: Set reasonable timeouts and recovery mechanisms\n"
    "   - STATE TRACKING: Keep track of app state to detect inconsistencies\n"
    "\n"
    "6. STUCK DETECTION OUTPUT:\n"
    "   - Include stuck_analysis field in JSON response\n"
    "   - Indicate if system appears stuck (true/false)\n"
    "   - Specify stuck type: vision_only, action_only, combined, or none\n"
    "   - Provide stuck indicators and evidence\n"
    "   - Suggest recovery actions and alternative strategies\n"
    "   - Recommend whether to continue, retry, or reset the automation\n"
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_messages(
    state: dict[str, Any],
    task_description: str,
    action_history: list[str],
    *,
    max_elements: int = 25,
    analysis_type: str = "comprehensive",
) -> list[dict[str, str]]:
    """Return list of messages ready for OpenAI client with enhanced element detection.
    
    Args:
        state: Current device state
        task_description: Current automation task
        action_history: Previous actions performed
        max_elements: Maximum elements to include in prompt
        analysis_type: Type of analysis to perform ("comprehensive", "element_detection", "context_analysis")
    """
    user_context = serialize_state(
        state,
        task_description,
        action_history,
        max_elements=max_elements,
    )
    
    # Add specialized prompts based on analysis type
    if analysis_type == "element_detection":
        user_prompt = f"{user_context}\n\n{_INSTRUCTIONS}\n{_ELEMENT_DETECTION_PROMPT}\n{_FOREGROUND_RETENTION_PROMPT}\n{_KEYBOARD_INTERACTION_PROMPT}\n{_STUCK_DETECTION_PROMPT}"
    elif analysis_type == "context_analysis":
        user_prompt = f"{user_context}\n\n{_INSTRUCTIONS}\n{_CONTEXT_ANALYSIS_PROMPT}\n{_FOREGROUND_RETENTION_PROMPT}\n{_KEYBOARD_INTERACTION_PROMPT}\n{_STUCK_DETECTION_PROMPT}"
    else:  # comprehensive
        user_prompt = f"{user_context}\n\n{_INSTRUCTIONS}\n{_ELEMENT_DETECTION_PROMPT}\n{_CONTEXT_ANALYSIS_PROMPT}\n{_FOREGROUND_RETENTION_PROMPT}\n{_KEYBOARD_INTERACTION_PROMPT}\n{_STUCK_DETECTION_PROMPT}"
    
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_element_detection_prompt(
    state: dict[str, Any],
    task_description: str,
    action_history: list[str],
    *,
    max_elements: int = 25,
) -> list[dict[str, str]]:
    """Build a specialized prompt focused on element detection and classification."""
    return build_messages(
        state, task_description, action_history, 
        max_elements=max_elements, analysis_type="element_detection"
    )


def build_context_analysis_prompt(
    state: dict[str, Any],
    task_description: str,
    action_history: list[str],
    *,
    max_elements: int = 25,
) -> list[dict[str, str]]:
    """Build a specialized prompt focused on context analysis and user flow understanding."""
    return build_messages(
        state, task_description, action_history, 
        max_elements=max_elements, analysis_type="context_analysis"
    )


def build_foreground_retention_prompt(
    state: dict[str, Any],
    task_description: str,
    action_history: list[str],
    *,
    max_elements: int = 25,
) -> list[dict[str, str]]:
    """Build a specialized prompt focused on keeping the app in foreground during automation."""
    user_context = serialize_state(
        state,
        task_description,
        action_history,
        max_elements=max_elements,
    )
    
    # Create a focused prompt for foreground retention
    foreground_instructions = (
        f"{user_context}\n\n"
        "SMART FOREGROUND MANAGEMENT: The app has automatic foreground recovery capabilities. "
        "You can interact with elements that temporarily leave the app, but avoid those that cause permanent loss.\n\n"
        f"{_INSTRUCTIONS}\n{_FOREGROUND_RETENTION_PROMPT}\n{_STUCK_DETECTION_PROMPT}\n"
        "ADDITIONAL FOREGROUND RETENTION RULES:\n"
        "1. AVOID elements that might cause permanent app closure or uninstallation\n"
        "2. ALLOW interaction with elements that temporarily leave the app (camera, share, settings)\n"
        "3. PRIORITIZE form completion and in-app navigation\n"
        "4. PREFER in-app alternatives when available\n"
        "5. FOCUS on core app functionality but don't avoid useful external features\n"
        "6. If unsure about an element's risk, evaluate if it returns to the app\n"
        "7. Consider the app's recovery capabilities when making decisions\n"
        "8. Balance functionality exploration with foreground stability\n"
    )
    
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": foreground_instructions},
    ]


def build_keyboard_interaction_prompt(
    state: dict[str, Any],
    task_description: str,
    action_history: list[str],
    *,
    max_elements: int = 25,
) -> list[dict[str, str]]:
    """Build a specialized prompt focused on keyboard detection and text input automation."""
    user_context = serialize_state(
        state,
        task_description,
        action_history,
        max_elements=max_elements,
    )
    
    # Create a focused prompt for keyboard interaction
    keyboard_instructions = (
        f"{user_context}\n\n"
        "KEYBOARD INTERACTION FOCUS: The app has text input fields that require keyboard interaction. "
        "Your primary goal is to detect the on-screen keyboard and execute precise taps to input text.\n\n"
        f"{_INSTRUCTIONS}\n{_KEYBOARD_INTERACTION_PROMPT}\n{_STUCK_DETECTION_PROMPT}\n"
        "ADDITIONAL KEYBOARD INTERACTION RULES:\n"
        "1. ALWAYS look for on-screen keyboard when text input fields are present\n"
        "2. TAP on input field first to focus and show keyboard\n"
        "3. IDENTIFY keyboard layout (QWERTY, numbers, symbols)\n"
        "4. TAP individual keys in sequence to type text\n"
        "5. USE space bar between words\n"
        "6. TAP Enter/Submit when done typing\n"
        "7. GENERATE realistic text based on field type (email, password, name, etc.)\n"
        "8. HANDLE special characters by switching to symbol keyboard\n"
        "9. USE backspace to correct typing mistakes\n"
        "10. PRIORITIZE text input completion over other actions\n"
    )
    
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": keyboard_instructions},
    ]


def build_action_prioritization_prompt(
    ui_elements: list[dict[str, Any]],
    task_description: str,
    action_history: list[dict[str, Any]],
    *,
    max_elements: int = 25,
    screenshot_with_boxes_base64: str | None = None,
) -> list[dict[str, str]]:
    """Build a prompt for action prioritization analysis.
    
    Args:
        ui_elements: List of UI elements from vision analysis
        task_description: Current automation task description
        action_history: History of previous actions performed
        max_elements: Maximum number of elements to include in analysis
        screenshot_with_boxes_base64: Base64 encoded screenshot with bounding boxes drawn
        
    Returns:
        List of message dictionaries for OpenAI API
    """
    # Limit elements for analysis
    elements_to_analyze = ui_elements[:max_elements]
    
    # Create state representation
    state = {
        "ui_elements": elements_to_analyze,
        "task_description": task_description,
        "action_history": action_history[-5:],  # Last 5 actions for context
        "timestamp": "current"
    }
    
    # Build the prompt
    system_message = {
        "role": "system",
        "content": (
            "You are an expert action prioritization system for Android automation. "
            "Your role is to analyze UI elements and determine the optimal next action "
            "based on order of precedence rules, task context, and exploration history.\n\n"
            
            "CRITICAL ORDER OF PRECEDENCE RULES (ALWAYS FOLLOW):\n"
            "1. SET_TEXT Action (Highest Priority):\n"
            "   - ALWAYS prioritize filling empty or relevant text input fields\n"
            "   - Choose first available empty/partially filled input field\n"
            "   - Keywords: email, password, search, name, address, phone, date, url, code\n"
            "   - Provide contextually appropriate dummy data\n\n"
            
            "2. TAP on Primary/Forwarding Elements (High Priority):\n"
            "   - Tap elements that advance user flow or reveal features\n"
            "   - Prioritize: Next, Continue, Sign In, Submit, Add to Cart, Save, Confirm\n"
            "   - Important links: View Details, Learn More, non-legal links\n"
            "   - Interactive controls: Unchecked checkboxes, radio buttons, switches\n\n"
            
            "3. SCROLL Action (Medium Priority):\n"
            "   - Only if no higher-priority actions available\n"
            "   - When screen is scrollable or content is truncated\n"
            "   - Prioritize: down → right → up → left\n\n"
            
            "4. TAP on Navigation/Secondary/Dismiss/Back Elements (Lowest Priority):\n"
            "   - Only when all higher-priority actions exhausted\n"
            "   - Purpose: Exit pop-ups, return to previous state, navigate side menus\n"
            "   - Prioritize: general navigation before Back/Dismiss\n\n"
            
            "EXPLORATION STRATEGY:\n"
            "• Prioritize unexplored elements over previously interacted ones\n"
            "• Consider element confidence and visibility\n"
            "• Balance task completion with exploration\n"
            "• Avoid repetitive actions that don't advance the task\n\n"
            
            "VISUAL ANALYSIS:\n"
            "• Use the provided screenshot with bounding boxes to understand element layout\n"
            "• Color-coded boxes: Green=buttons, Blue=inputs, Yellow=templates, Red=text\n"
            "• Analyze element positioning and relationships\n"
            "• Consider visual hierarchy and user flow\n\n"
            
            "EXCLUSION RULES (STRICT):\n"
            "• NEVER tap buttons that start third-party authentication such as 'Continue with Google',\n"
            "  'Sign in with Facebook', 'Apple', 'Twitter', 'LinkedIn', or 'Microsoft'.\n"
            "  Skip any element whose text includes these provider names unless the task explicitly\n"
            "  instructs otherwise.\n\n"
            
            "OUTPUT FORMAT:\n"
            "Provide a JSON response with:\n"
            "- suggestions: List of suggested actions with confidence scores\n"
            "- confidence: Overall confidence in the analysis (0.0-1.0)\n"
            "- reasoning: Explanation of the prioritization logic\n"
            "- element_priorities: Priority scores for each element\n"
        )
    }
    
    # Create user message with current state
    user_content = (
        f"TASK: {task_description}\n\n"
        f"UI ELEMENTS ({len(elements_to_analyze)} found):\n"
    )
    
    for i, element in enumerate(elements_to_analyze):
        text = element.get('text', 'No text')
        element_type = element.get('element_type', 'Unknown')
        bounds = element.get('bounds', {})
        x, y = bounds.get('x', 0), bounds.get('y', 0)
        
        user_content += (
            f"{i+1}. Text: '{text}' | Type: {element_type} | "
            f"Position: ({x}, {y})\n"
        )
    
    if action_history:
        user_content += f"\nRECENT ACTIONS ({len(action_history)}):\n"
        for action in action_history[-3:]:  # Last 3 actions
            action_type = action.get('type', 'Unknown')
            element_text = action.get('element_text', 'Unknown')
            user_content += f"• {action_type}: '{element_text}'\n"
    
    user_content += (
        "\nANALYSIS REQUEST:\n"
        "Based on the order of precedence rules and current context, "
        "determine the optimal next action. Consider:\n"
        "1. Which elements match the highest priority action type?\n"
        "2. Which elements are unexplored and should be prioritized?\n"
        "3. What action will best advance the current task?\n"
        "4. How confident are you in the element detection and classification?\n\n"
        
        "Provide your analysis in JSON format with suggestions, confidence, reasoning, and element priorities."
    )
    
    user_message = {
        "role": "user",
        "content": user_content
    }
    
    # Add image if available
    messages = [system_message]
    
    if screenshot_with_boxes_base64:
        # Add image message for visual analysis using base64
        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the current screen with bounding boxes drawn around detected UI elements. Use this visual information to enhance your analysis."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{screenshot_with_boxes_base64}",
                        "detail": "high"
                    }
                }
            ]
        }
        messages.append(image_message)
    
    messages.append(user_message)
    
    return messages


__all__ = ["build_messages", "build_element_detection_prompt", "build_context_analysis_prompt", "build_foreground_retention_prompt", "build_keyboard_interaction_prompt", "build_action_prioritization_prompt"] 