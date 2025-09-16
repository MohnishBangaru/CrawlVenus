# Phi Ground Integration with AA_VA

This document describes the integration of Phi Ground with AA_VA (Android Automation Vision AI) framework, implementing the exact approach from the paper "Phi Ground: A Framework for Learning Grounded Policy with Large Language Models" for touch action generation instead of mouse actions.

## Overview

Phi Ground is a vision-language model that can understand screen content and generate appropriate actions. This integration follows the paper's methodology:

1. **Screen Understanding**: Phi Ground analyzes screenshots to understand the current UI state
2. **Action Generation**: Based on screen content and task description, it generates appropriate touch actions
3. **Touch Coordinate Prediction**: Instead of mouse actions, it predicts touch coordinates for Android automation

## Architecture

### Core Components

1. **PhiGroundActionGenerator** (`src/ai/phi_ground.py`)
   - Main class for Phi Ground integration
   - Handles model initialization and action generation
   - Follows the paper's prompt format and response parsing

2. **Enhanced Action Determiner** (`src/ai/action_determiner.py`)
   - Integrates Phi Ground as the primary action generation method
   - Falls back to traditional methods if Phi Ground fails
   - Maintains backward compatibility

3. **Action Prioritizer** (`src/core/action_prioritizer.py`)
   - Uses Phi Ground for optimal action selection
   - Validates generated actions and coordinates
   - Provides confidence scoring

### Integration Flow

```
Screenshot Capture → Phi Ground Analysis → Action Generation → Validation → Execution
```

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# Phi Ground Configuration
USE_PHI_GROUND=true
PHI_GROUND_MODEL=microsoft/Phi-3-vision-128k-instruct
PHI_GROUND_TEMPERATURE=0.7
PHI_GROUND_MAX_TOKENS=256
PHI_GROUND_CONFIDENCE_THRESHOLD=0.5
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_phi_ground` | `true` | Enable/disable Phi Ground integration |
| `phi_ground_model` | `microsoft/Phi-3-vision-128k-instruct` | Phi Ground model to use |
| `phi_ground_temperature` | `0.7` | Temperature for action generation |
| `phi_ground_max_tokens` | `256` | Maximum tokens for generation |
| `phi_ground_confidence_threshold` | `0.5` | Minimum confidence for actions |

## Usage

### Basic Usage

```python
from src.ai.phi_ground import get_phi_ground_generator
from src.vision.models import UIElement, BoundingBox

# Initialize Phi Ground
phi_ground = get_phi_ground_generator()
await phi_ground.initialize()

# Generate action
action = await phi_ground.generate_touch_action(
    screenshot_path="screenshot.png",
    task_description="Login to the application",
    action_history=[{"type": "tap", "element_text": "Login"}],
    ui_elements=[...]  # List of UIElement objects
)

if action:
    print(f"Generated action: {action['type']}")
    print(f"Coordinates: ({action['x']}, {action['y']})")
    print(f"Confidence: {action['confidence']}")
```

### Integration with Action Determiner

```python
from src.ai.action_determiner import get_enhanced_action_determiner

# Get action determiner (automatically uses Phi Ground if enabled)
determiner = get_enhanced_action_determiner()

# Determine next action (Phi Ground will be used automatically)
action = await determiner.determine_next_action(
    ui_elements=ui_elements,
    task_description=task_description,
    action_history=action_history,
    device_info=device_info,
    screenshot_path=screenshot_path  # Required for Phi Ground
)
```

### Integration with Action Prioritizer

```python
from src.core.action_prioritizer import get_action_prioritizer

# Get action prioritizer
prioritizer = get_action_prioritizer()

# Get optimal action (Phi Ground will be used if screenshot available)
optimal_action = prioritizer.get_optimal_action(
    ui_elements=ui_elements,
    task_description=task_description,
    action_history=action_history,
    screenshot_path=screenshot_path  # Required for Phi Ground
)
```

## Action Types

Phi Ground generates the following action types:

### 1. TAP Actions
```json
{
    "type": "tap",
    "x": 200,
    "y": 300,
    "element_text": "Login",
    "reasoning": "Phi Ground: TAP: Login button at coordinates (200, 300)",
    "phi_ground_generated": true,
    "confidence": 0.85
}
```

### 2. TEXT_INPUT Actions
```json
{
    "type": "text_input",
    "x": 150,
    "y": 400,
    "text": "user@example.com",
    "field_hint": "Email",
    "reasoning": "Phi Ground: INPUT: \"user@example.com\" into Email field at coordinates (150, 400)",
    "phi_ground_generated": true,
    "confidence": 0.9
}
```

### 3. SWIPE Actions
```json
{
    "type": "swipe",
    "x1": 500,
    "y1": 800,
    "x2": 500,
    "y2": 200,
    "duration": 300,
    "reasoning": "Phi Ground: SWIPE: from (500, 800) to (500, 200)",
    "phi_ground_generated": true
}
```

### 4. WAIT Actions
```json
{
    "type": "wait",
    "duration": 2.0,
    "reasoning": "Phi Ground: WAIT: 2 seconds",
    "phi_ground_generated": true
}
```

## Prompt Format

Phi Ground uses the following prompt format (following the paper):

```
<|im_start|>system
You are an Android automation assistant. Analyze the screenshot and generate appropriate touch actions to accomplish the given task. Focus on the app interface, not system UI elements.

Task: [task_description]
Recent actions: [action_history]

Generate actions in this format:
1. TAP: [element_description] at coordinates (x, y)
2. INPUT: [text] into [field_description] at coordinates (x, y)
3. SWIPE: from (x1, y1) to (x2, y2)
4. WAIT: [duration] seconds

Only generate one action at a time. Prioritize:
1. Text input fields that need to be filled
2. Primary action buttons (Continue, Submit, Next, etc.)
3. Interactive elements (buttons, links, etc.)
4. Navigation elements as last resort

Coordinates should be within the app area (avoid status bar, navigation bar).
<|im_end|>
<|im_start|>user
<|image|>
Please analyze this Android app screenshot and suggest the next touch action to accomplish the task.
<|im_end|>
<|im_start|>assistant
```

## Validation

### Coordinate Validation
- Ensures coordinates are within screen bounds
- Validates action parameters (x, y for taps, x1, y1, x2, y2 for swipes)
- Checks for reasonable coordinate ranges

### Confidence Threshold
- Actions below the confidence threshold are rejected
- Falls back to traditional methods if confidence is too low
- Configurable threshold in settings

### Element Matching
- Matches Phi Ground descriptions to detected UI elements
- Uses text similarity and position information
- Provides confidence scores based on matching quality

## Fallback Strategy

If Phi Ground fails or generates invalid actions, the system falls back to traditional methods:

1. **LLM Analysis**: Uses OpenAI GPT for action determination
2. **Rule-based Selection**: Applies predefined rules and heuristics
3. **Element Prioritization**: Uses confidence scores and exploration history

## Testing

### Run Integration Tests

```bash
# Test Phi Ground integration
python scripts/test_phi_ground.py

# Test with specific APK
python scripts/universal_apk_tester.py --apk path/to/app.apk --actions 10
```

### Test Configuration

```bash
# Disable Phi Ground for comparison
export USE_PHI_GROUND=false

# Enable Phi Ground with custom settings
export PHI_GROUND_MODEL=microsoft/Phi-3-vision-128k-instruct
export PHI_GROUND_CONFIDENCE_THRESHOLD=0.7
```

## Performance Considerations

### Model Loading
- Phi Ground model is loaded once and reused
- Uses GPU if available, falls back to CPU
- Model size: ~8GB (Phi-3-vision-128k-instruct)

### Generation Speed
- Typical generation time: 2-5 seconds
- Depends on hardware (GPU recommended)
- Cached for repeated screenshots

### Memory Usage
- Model requires significant RAM (8-16GB recommended)
- Uses torch.float16 on GPU for efficiency
- Automatic memory management

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check internet connection for model download
   - Ensure sufficient disk space (~8GB)
   - Verify CUDA installation for GPU support

2. **Low Confidence Actions**
   - Adjust `phi_ground_confidence_threshold`
   - Check screenshot quality and resolution
   - Verify UI element detection

3. **Invalid Coordinates**
   - Check screen resolution settings
   - Verify coordinate validation logic
   - Ensure screenshot is properly captured

4. **Fallback to Traditional Methods**
   - Check Phi Ground configuration
   - Verify screenshot path is provided
   - Review error logs for specific issues

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("src.ai.phi_ground").setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Model Fine-tuning**: Fine-tune Phi Ground on Android UI datasets
2. **Multi-modal Input**: Combine screenshot with accessibility tree data
3. **Action Sequences**: Generate multi-step action sequences
4. **Context Awareness**: Better understanding of app state and flow
5. **Performance Optimization**: Model quantization and optimization

## References

- [Phi Ground Paper](https://arxiv.org/pdf/2507.23779)
- [Phi-3 Vision Model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
- [AA_VA Framework Documentation](README.md)

## License

This integration follows the same license as the AA_VA framework and respects the Phi Ground model's licensing terms.
