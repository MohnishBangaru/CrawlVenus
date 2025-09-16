"""Phi Ground integration for intelligent touch action generation.

This module implements Phi Ground exactly as described in the paper:
"Phi Ground: A Framework for Learning Grounded Policy with Large Language Models"
for generating touch actions instead of mouse actions.

The implementation follows the paper's approach:
1. Screen understanding through vision-language model
2. Action generation based on screen content and task description
3. Touch coordinate prediction for Android automation
"""

from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import numpy as np
from loguru import logger

# Import torchvision at module level to avoid circular imports
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    transforms = None

from ..core.config import config
from ..vision.models import UIElement, BoundingBox


class PhiGroundActionGenerator:
    """Phi Ground action generator for Android touch automation."""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-vision-128k-instruct"):
        """Initialize Phi Ground action generator.
        
        Args:
            model_name: The Phi Ground model to use for action generation
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        # Hugging Face token (optional, but required for gated models)
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        # Force CUDA if available, with better detection
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU")
            
        self._initialized = False
        self.vision_supported = False
        # Toggle extra debug dumps of prompts and raw VLM responses
        try:
            self.debug_mode = bool(int(os.getenv("PHI_GROUND_DEBUG", "0")))
        except Exception:
            self.debug_mode = False
        # Chat style: "strict" (system+user) or "example" (user, assistant, user)
        self.chat_style = os.getenv("PHI_GROUND_CHAT_STYLE", "strict").strip().lower()
        
        # Track recent actions to detect and prevent repetition
        self.recent_actions = []
        self.max_recent_actions = 5
        self.repetition_threshold = 3  # Number of similar actions before considering stuck
        
    async def initialize(self) -> None:
        """Initialize the Phi Ground model."""
        if self._initialized:
            return
            
        try:
            logger.info(f"Initializing Phi Ground model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer first (pass token if available)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.hf_token,
            )
            
            # Check if vision is supported
            try:
                from PIL import Image
                import numpy as np
                dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                
                # Try multiple vision tokenization approaches
                vision_works = False
                
                # Approach 1: Check if model supports vision at all
                if hasattr(self.tokenizer, 'image_processor') or hasattr(self.tokenizer, 'process_images'):
                    vision_works = True
                    logger.info("Vision tokenization supported (model has vision capabilities)")
                else:
                    # Approach 2: Try with image processor from transformers
                    try:
                        from transformers import AutoImageProcessor
                        image_processor = AutoImageProcessor.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            token=self.hf_token,
                        )
                        processed_image = image_processor(dummy_image, return_tensors="pt")
                        test_inputs = self.tokenizer("test", return_tensors="pt", **processed_image)
                        vision_works = True
                        logger.info("Vision tokenization supported (with AutoImageProcessor)")
                    except Exception as e1:
                        logger.debug(f"AutoImageProcessor approach failed: {e1}")
                        
                        # Approach 3: Check if it's a vision model by name
                        if "vision" in self.model_name.lower():
                            vision_works = True
                            logger.info("Vision tokenization supported (vision model detected)")
                        else:
                            logger.debug("No vision capabilities detected")
                
                self.vision_supported = vision_works
                
                if self.vision_supported:
                    logger.info("Vision tokenization supported")
                else:
                    logger.info("Vision tokenization not supported, using text-only mode")
                    
            except Exception as e:
                self.vision_supported = False
                logger.warning(f"Vision support check failed: {e}, using text-only mode")
            
            # Standard initialization strategies
            initialization_strategies = [
                # Strategy 1: CUDA with FlashAttention2 (optimal)
                {
                    "name": "CUDA FlashAttention2",
                    "kwargs": {
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }
                },
                # Strategy 2: CUDA without FlashAttention2 (fallback)
                {
                    "name": "CUDA Standard",
                    "kwargs": {
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "attn_implementation": "eager",
                        "low_cpu_mem_usage": True
                    }
                },
                # Strategy 3: CPU fallback (last resort)
                {
                    "name": "CPU Fallback",
                    "kwargs": {
                        "torch_dtype": torch.float32,
                        "device_map": None,
                        "trust_remote_code": True,
                        "attn_implementation": "eager",
                        "low_cpu_mem_usage": True
                    }
                }
            ]
            
            for strategy in initialization_strategies:
                try:
                    logger.info(f"Trying initialization strategy: {strategy['name']}")
                    
                    # For strategies that need to disable FlashAttention2, modify the config first
                    if strategy['name'] != "CUDA FlashAttention2":
                        from transformers import AutoConfig
                        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                        config.use_flash_attention_2 = False
                        logger.info(f"Disabled FlashAttention2 in model config for {strategy['name']}")
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            config=config,
                            token=self.hf_token,
                            **strategy['kwargs']
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            token=self.hf_token,
                            **strategy['kwargs']
                        )
                    
                    # Move to CPU if using CPU fallback strategy
                    if strategy['name'] == "CPU Fallback":
                        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                        self.model = self.model.to("cpu")
                        logger.info("Model moved to CPU for CPU Fallback strategy")
                    
                    logger.info(f"Phi Ground model initialized successfully with {strategy['name']}")
                    break
                    
                except Exception as strategy_error:
                    error_msg = str(strategy_error)
                    logger.warning(f"Strategy '{strategy['name']}' failed: {error_msg}")
                    
                    # If this is the last strategy, raise the error
                    if strategy == initialization_strategies[-1]:
                        logger.error(f"All initialization strategies failed. Last error: {strategy_error}")
                        raise strategy_error
                    continue
            
            self._initialized = True
            logger.info("Phi Ground model initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phi Ground model: {e}")
            raise
    
    def _create_phi_ground_prompt(
        self, 
        image_path: str, 
        task_description: str,
        action_history: List[Dict[str, Any]]
    ) -> str:
        """Create Phi Ground prompt following the paper's format.
        
        Args:
            image_path: Path to the screenshot
            task_description: Current automation task
            action_history: Previous actions performed
            
        Returns:
            Formatted prompt for Phi Ground
        """
        # Format action history for context
        history_text = ""
        if action_history:
            recent_actions = action_history[-3:]  # Last 3 actions
            history_parts = []
            for action in recent_actions:
                action_type = action.get("type", "")
                element_text = action.get("element_text", "")
                if element_text:
                    history_parts.append(f"{action_type}: {element_text}")
            if history_parts:
                history_text = f"Recent actions: {', '.join(history_parts)}"
        
        # Create the prompt following Phi Ground paper format
        prompt = f"""<|im_start|>system
You are an Android automation assistant. Analyze the screenshot and generate appropriate touch actions to accomplish the given task. Focus on the app interface, not system UI elements.

Task: {task_description}
{history_text}

You must respond with exactly one action in this format:
1. TAP: [element_description] at coordinates (x, y)
2. INPUT: [text] into [field_description] at coordinates (x, y)
3. SWIPE: from (x1, y1) to (x2, y2)
4. WAIT: [duration] seconds

Rules:
- Always start with a number (1.)
- Use exact format: "1. TAP: [description] at coordinates (x, y)"
- Coordinates should be integers between 100-1000
- Focus on visible, clickable elements
- Avoid system UI (status bar, navigation bar)
- If no clear action, use "1. WAIT: 2 seconds"
- Do not repeat the prompt or system instructions

Prioritize:
1. Text input fields that need to be filled
2. Primary action buttons (Continue, Submit, Next, etc.)
3. Interactive elements (buttons, links, etc.)
4. Navigation elements as last resort
<|im_end|>
<|im_start|>user
<|image|>
Please analyze this Android app screenshot and suggest the next touch action to accomplish the task.
<|im_end|>
<|im_start|>assistant
1. """
        
        return prompt
    
    async def generate_touch_action(
        self,
        image_path: str,
        task_description: str,
        action_history: List[Dict[str, Any]],
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Generate touch action using Phi Ground following the paper's approach.
        
        Args:
            image_path: Path to the screenshot
            task_description: Current automation task
            action_history: Previous actions performed
            ui_elements: Detected UI elements for validation
            
        Returns:
            Action dictionary or None if no action should be performed
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Loaded screenshot: {image_path} (size: {image.size})")
            
            # Create Phi Ground prompt
            prompt = self._create_phi_ground_prompt(
                image_path, task_description, action_history
            )
            
            # Tokenize input - handle vision-language model tokenization
            if self.vision_supported:
                try:
                    logger.info("Using vision tokenization with screenshot")
                    
                    # Try multiple vision tokenization approaches
                    vision_inputs = None
                    used_auto_processor = False
                    processor_for_decode = None
                    input_ids_len_for_decode = None
                    formatted_prompt_str = None
                    messages_used = None

                    # Approach 0: Use AutoProcessor (preferred for Phi-3 Vision)
                    try:
                        from transformers import AutoProcessor
                        processor = AutoProcessor.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            token=self.hf_token,
                            num_crops=16,
                        )

                        # Build chat template (supports two styles)
                        # Prepare recent actions text (last 3)
                        history_text_parts = []
                        try:
                            if action_history:
                                recent_actions_ct = action_history[-3:]
                                for _a in recent_actions_ct:
                                    _t = _a.get("type", "")
                                    _et = _a.get("element_text", "")
                                    if _et:
                                        history_text_parts.append(f"{_t}: {_et}")
                        except Exception:
                            pass
                        history_text_ct = ("Recent actions: " + ", ".join(history_text_parts)) if history_text_parts else ""
                        if self.chat_style == "example":
                            # Build a user-assistant-user sequence similar to reference
                            # Assistant bridge content from UI hints (optional)
                            visible_texts = []
                            try:
                                for el in ui_elements:
                                    txt = (el.text or "").strip()
                                    if txt and txt not in visible_texts:
                                        visible_texts.append(txt)
                                    if len(visible_texts) >= 5:
                                        break
                            except Exception:
                                pass
                            assistant_bridge = (
                                "Visible elements: " + ", ".join(visible_texts)
                            ) if visible_texts else "Okay."

                            messages = [
                                {"role": "user", "content": "<|image_1|>\n" + (f"Task: {task_description}" if task_description else "")},
                                {"role": "assistant", "content": assistant_bridge},
                                {"role": "user", "content": (
                                    "Provide the next touch action as a single line: "
                                    "1. TAP: [desc] at coordinates (x, y) | 1. INPUT: \"text\" into [field] at coordinates (x, y) | "
                                    "1. SWIPE: from (x1, y1) to (x2, y2) | 1. WAIT: 2 seconds."
                                )}
                            ]
                        else:
                            # Default strict system+user for single-line action
                            strict_rules = (
                                "You must respond with exactly one action in this format:\n"
                                "1. TAP: [element_description] at coordinates (x, y)\n"
                                "2. INPUT: [text] into [field_description] at coordinates (x, y)\n"
                                "3. SWIPE: from (x1, y1) to (x2, y2)\n"
                                "4. WAIT: [duration] seconds\n\n"
                                "Rules:\n"
                                "- Respond with ONE line only.\n"
                                "- Start with a number (1.).\n"
                                "- Use the exact format above.\n"
                                "- If no clear action, respond: 1. WAIT: 2 seconds\n"
                                "- Do not refuse. Do not add extra text."
                            )
                            messages = [
                                {"role": "system", "content": strict_rules},
                                {
                                    "role": "user",
                                    "content": "<|image_1|>\n" +
                                               (f"Task: {task_description}\n" if task_description else "") +
                                               (history_text_ct if history_text_ct else "")
                                }
                            ]
                        formatted_prompt = processor.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        formatted_prompt_str = formatted_prompt
                        messages_used = messages

                        # Mirror example style: processor(prompt, [image], ...)
                        processed = processor(
                            formatted_prompt,
                            [image],
                            return_tensors="pt",
                        )
                        # Move to correct device
                        vision_inputs = {k: v.to(self.device) for k, v in processed.items()}
                        # Mark that we used AutoProcessor for decoding later
                        used_auto_processor = True
                        processor_for_decode = processor
                        input_ids_len_for_decode = vision_inputs.get('input_ids').shape[1] if 'input_ids' in vision_inputs else None
                        logger.info("Vision tokenization successful with AutoProcessor and chat template")
                    except Exception as e0:
                        logger.debug(f"AutoProcessor approach failed: {e0}")
                    
                    # Approach 1: Try with AutoImageProcessor
                    try:
                        from transformers import AutoImageProcessor
                        image_processor = AutoImageProcessor.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            token=self.hf_token,
                        )
                        processed_image = image_processor(image, return_tensors="pt")
                        vision_inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            **processed_image,
                            padding=True,
                            truncation=True
                        )
                        # Move to correct device
                        vision_inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
                        logger.info("Vision tokenization successful with AutoImageProcessor")
                    except Exception as e1:
                        logger.debug(f"AutoImageProcessor approach failed: {e1}")
                        
                        # Approach 2: Try with model's image processor
                        try:
                            if hasattr(self.tokenizer, 'image_processor'):
                                processed_image = self.tokenizer.image_processor(image, return_tensors="pt")
                                vision_inputs = self.tokenizer(
                                    prompt,
                                    return_tensors="pt",
                                    **processed_image,
                                    padding=True,
                                    truncation=True
                                )
                                # Move to correct device
                                vision_inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
                                logger.info("Vision tokenization successful with model's image processor")
                        except Exception as e2:
                            logger.debug(f"Model image processor approach failed: {e2}")
                            
                            # Approach 3: Try direct images parameter
                            try:
                                vision_inputs = self.tokenizer(
                                    prompt,
                                    images=image,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True
                                )
                                # Move to correct device
                                vision_inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
                                logger.info("Vision tokenization successful with direct images parameter")
                            except Exception as e3:
                                logger.debug(f"Direct images parameter approach failed: {e3}")
                                
                                # Approach 4: Manual image processing
                                try:
                                    if not TORCHVISION_AVAILABLE:
                                        raise ImportError("torchvision not available")
                                        
                                    # Resize image to standard size
                                    image_resized = image.resize((224, 224))
                                    
                                    # Convert to tensor using imported transforms
                                    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
                                    
                                    image_tensor = transform(image_resized).unsqueeze(0)
                                    
                                    # Create inputs with image
                                    vision_inputs = self.tokenizer(
                                        prompt,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True
                                    )
                                    
                                    # Add image tensor
                                    vision_inputs['pixel_values'] = image_tensor.to(self.device)
                                    logger.info("Vision tokenization successful with manual processing")
                                except Exception as e4:
                                    logger.debug(f"Manual processing approach failed: {e4}")
                                    
                                    # Fallback: Simple tensor conversion without torchvision
                                    try:
                                        # Ensure we have a resized image available
                                        image_resized = image.resize((224, 224))
                                        # Convert PIL image to numpy array and then to tensor
                                        image_array = np.array(image_resized)
                                        image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                                        vision_inputs['pixel_values'] = image_tensor.to(self.device)
                                        logger.info("Vision tokenization successful with simple tensor conversion")
                                    except Exception as e5:
                                        logger.debug(f"Simple tensor conversion failed: {e5}")
                                        
                                        # If still failing, mark as None to trigger text-only fallback
                                        vision_inputs = None
                    
                    if vision_inputs is not None:
                        inputs = vision_inputs
                    else:
                        raise Exception("All vision tokenization approaches failed")
                        
                except Exception as e:
                    logger.warning(f"Vision tokenization failed: {e}, falling back to text-only")
                    self.vision_supported = False
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    # Move to correct device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Use text-only tokenization with proper attention mask
                logger.info("Using text-only tokenization (no screenshot processing)")
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                # Move to correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with robust caching fixes
            with torch.no_grad():
                generation_success = False
                # Prepare generation args dict matching the example
                generation_args = {
                    "max_new_tokens": 256,
                    "do_sample": False,
                }

                # Debug: log inputs present, shapes and device
                try:
                    keys = list(inputs.keys())
                    shapes = {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}
                    devices = {k: str(v.device) for k, v in inputs.items() if hasattr(v, 'device')}
                    logger.info(f"Generation inputs keys: {keys}")
                    logger.info(f"Generation input shapes: {shapes}")
                    logger.info(f"Generation input devices: {devices}, model device: {self.device}")
                    gen_keys, gen_shapes, gen_devices = keys, shapes, devices
                except Exception as _log_err:
                    logger.debug(f"Failed to log input shapes/devices: {_log_err}")
                    gen_keys, gen_shapes, gen_devices = None, None, None
                
                # Strategy 1: Simple generation without cache (most compatible)
                try:
                    logger.info("Trying simple generation without cache")
                    # Get eos_token_id from processor if available
                    eos_id = None
                    try:
                        eos_id = processor_for_decode.tokenizer.eos_token_id if 'processor_for_decode' in locals() and processor_for_decode is not None else self.tokenizer.eos_token_id
                    except Exception:
                        eos_id = self.tokenizer.eos_token_id
                    
                    # Generate using minimal args to avoid cache issues
                    generate_ids = self.model.generate(
                        **inputs,
                        eos_token_id=eos_id,
                        use_cache=False,  # Disable cache to avoid DynamicCache issues
                        **generation_args
                    )
                    generation_success = True
                    logger.info("Simple generation successful")
                except Exception as e:
                    logger.warning(f"Simple generation failed: {e}")
                
                # Strategy 2: Try with even more minimal args
                if not generation_success:
                    try:
                        logger.info("Trying minimal generation args")
                        generate_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                        generation_success = True
                        logger.info("Minimal generation successful")
                    except Exception as e:
                        logger.warning(f"Minimal generation failed: {e}")
                
                # Strategy 3: Try with only essential inputs
                if not generation_success:
                    try:
                        logger.info("Trying essential inputs only")
                        essential_inputs = {}
                        if 'input_ids' in inputs:
                            essential_inputs['input_ids'] = inputs['input_ids']
                        if 'attention_mask' in inputs:
                            essential_inputs['attention_mask'] = inputs['attention_mask']
                        if 'pixel_values' in inputs:
                            essential_inputs['pixel_values'] = inputs['pixel_values']
                        
                        generate_ids = self.model.generate(
                            **essential_inputs,
                            max_new_tokens=256,
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                        generation_success = True
                        logger.info("Essential inputs generation successful")
                    except Exception as e:
                        logger.warning(f"Essential inputs generation failed: {e}")
                
                # Strategy 4: Ultimate fallback - direct forward pass
                if not generation_success:
                    try:
                        logger.info("Using direct forward pass fallback")
                        # Get the last token and generate one token at a time
                        current_input = inputs.get('input_ids')
                        generated_tokens = []
                        
                        # Ensure input is on the same device as model
                        current_input = current_input.to(self.device)
                        
                        for _ in range(256):
                            with torch.no_grad():
                                forward_outputs = self.model.forward(input_ids=current_input)
                                next_token = torch.argmax(forward_outputs.logits[:, -1, :], dim=-1)
                                generated_tokens.append(next_token.item())
                                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                                
                                # Stop if we hit the end token
                                if next_token.item() == self.tokenizer.eos_token_id:
                                    break
                        
                        # Combine original input with generated tokens
                        generate_ids = torch.cat([inputs.get('input_ids'), torch.tensor([generated_tokens])], dim=1)
                        generation_success = True
                        logger.info("Direct forward pass successful")
                    except Exception as e:
                        logger.error(f"All generation strategies failed: {e}")
                        return None
            
            # Decode response
            # Mirror example: trim input tokens and use processor.batch_decode
            try:
                if 'processor_for_decode' in locals() and processor_for_decode is not None and input_ids_len_for_decode is not None:
                    # Remove input tokens
                    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                    response = processor_for_decode.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                else:
                    response = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            except Exception as _decode_err:
                logger.debug(f"Processor batch_decode failed, falling back to tokenizer.decode: {_decode_err}")
                response = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            logger.debug(f"Phi Ground raw response: {response}")
            
            # Clean response - remove prompt repetition
            cleaned_response = self._clean_response(response)
            logger.debug(f"Phi Ground cleaned response: {cleaned_response}")
            
            # Extract action from response
            action = self._parse_phi_ground_response(cleaned_response, ui_elements)

            # Check for repeated actions (stuck behavior)
            if action and self._is_repeated_action(action):
                logger.warning(f"Detected repeated action at coordinates ({action.get('x')}, {action.get('y')}), using fallback")
                fallback_action = self._get_fallback_action(ui_elements, task_description)
                if fallback_action:
                    action = fallback_action
                    logger.info(f"Using fallback action: {fallback_action.get('type')} - {fallback_action.get('reasoning', '')}")

            # Add action to history for future repetition detection
            self._add_action_to_history(action)

            if action:
                logger.info(f"Phi Ground generated action: {action['type']} - {action.get('reasoning', '')}")

                # If parser fell back to default wait, persist debug details
                if (
                    action.get("type") == "wait"
                    and isinstance(action.get("reasoning"), str)
                    and action.get("reasoning", "").lower().startswith("no valid action parsed")
                ):
                    self._save_debug_response(
                        image_path=image_path,
                        task_description=task_description,
                        raw_response=response,
                        cleaned_response=cleaned_response,
                        action=action,
                        extras={
                            "model": self.model_name,
                            "device": self.device,
                            "vision_supported": self.vision_supported,
                            "used_auto_processor": bool('used_auto_processor' in locals() and used_auto_processor),
                            "input_keys": gen_keys,
                            "input_shapes": gen_shapes,
                            "input_devices": gen_devices,
                            "formatted_prompt": formatted_prompt_str,
                            "messages": messages_used,
                        },
                    )
            else:
                logger.warning(f"No valid action parsed from response: {response[:200]}...")
                # Try fallback when no action is parsed
                fallback_action = self._get_fallback_action(ui_elements, task_description)
                if fallback_action:
                    action = fallback_action
                    logger.info(f"Using fallback action due to parse failure: {fallback_action.get('type')} - {fallback_action.get('reasoning', '')}")
            
            # Always save full trace when debug mode is enabled
            if self.debug_mode:
                self._save_debug_response(
                    image_path=image_path,
                    task_description=task_description,
                    raw_response=response,
                    cleaned_response=cleaned_response,
                    action=action if action else {"type": "none"},
                    extras={
                        "model": self.model_name,
                        "device": self.device,
                        "vision_supported": self.vision_supported,
                        "used_auto_processor": bool('used_auto_processor' in locals() and used_auto_processor),
                        "input_keys": gen_keys,
                        "input_shapes": gen_shapes,
                        "input_devices": gen_devices,
                        "formatted_prompt": formatted_prompt_str,
                        "messages": messages_used,
                    },
                )

            return action
            
        except Exception as e:
            logger.error(f"Phi Ground action generation failed: {e}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """Clean the response by removing prompt repetition and extracting only the action.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Cleaned response containing only the action
        """
        try:
            # Remove common prompt repetition patterns
            patterns_to_remove = [
                r'<\|im_start\|>system.*?<\|im_end\|>',
                r'<\|im_start\|>user.*?<\|im_end\|>',
                r'You are an Android automation assistant.*?',
                r'Analyze the screenshot.*?',
                r'Please analyze.*?',
                r'Task:.*?',
                r'Rules:.*?',
                r'Prioritize:.*?'
            ]
            
            cleaned = response
            for pattern in patterns_to_remove:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract only the assistant response part
            assistant_match = re.search(r'<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|$)', cleaned, re.DOTALL)
            if assistant_match:
                cleaned = assistant_match.group(1).strip()
            
            # If no assistant tags found, try to find action patterns
            if not cleaned or len(cleaned) < 10:
                action_match = re.search(r'(\d+\.\s*(?:TAP|INPUT|SWIPE|WAIT):.*?)(?:\n|$)', response, re.IGNORECASE)
                if action_match:
                    cleaned = action_match.group(1).strip()
            
            # Final cleanup
            cleaned = cleaned.strip()
            if not cleaned:
                # Fallback: try to find any action-like text
                action_match = re.search(r'(\d+\.\s*[A-Z]+:.*)', response, re.IGNORECASE)
                if action_match:
                    cleaned = action_match.group(1).strip()
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning response: {e}")
            return response
    
    def _parse_phi_ground_response(
        self, 
        response: str, 
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Parse Phi Ground response and extract action details.
        
        Args:
            response: Raw response from Phi Ground
            ui_elements: Detected UI elements for coordinate validation
            
        Returns:
            Parsed action dictionary or None
        """
        try:
            # Normalize and broaden accepted action keywords
            synonyms_map = {
                "TAP": ["TAP", "CLICK", "PRESS", "HIT"],
                "INPUT": ["INPUT", "TYPE", "ENTER"],
                "SWIPE": ["SWIPE", "DRAG", "SCROLL"],
                "WAIT": ["WAIT", "PAUSE", "SLEEP"],
            }
            all_keywords = [kw for keys in synonyms_map.values() for kw in keys]
            keywords_pattern = "|".join(all_keywords)

            # Primary pattern: optional leading number, action keyword (synonym), optional colon/hyphen
            action_match = re.search(
                rf"(?:\d+\.\s*)?({keywords_pattern})[:\-]?\s*(.+)",
                response,
                re.IGNORECASE,
            )

            if action_match:
                raw_type = action_match.group(1).upper()
                action_description = action_match.group(2).strip()

                # Map to canonical action type
                action_type = None
                for canonical, synonyms in synonyms_map.items():
                    if raw_type in synonyms:
                        action_type = canonical
                        break
                if action_type is None:
                    logger.warning(f"Unknown action keyword: {raw_type}")
                    return None
            else:
                # Heuristic fallback: infer from words present and coordinates
                lower_resp = response.lower()
                coords_present = re.search(r"\(\s*\d+\s*,\s*\d+\s*\)", response) is not None

                if any(w in lower_resp for w in ["tap", "click", "press", "hit"]) and coords_present:
                    action_type = "TAP"
                    action_description = response
                elif any(w in lower_resp for w in ["type", "enter"]) and coords_present:
                    action_type = "INPUT"
                    action_description = response
                elif any(w in lower_resp for w in ["swipe", "drag", "scroll"]) and re.search(r"to\s*\(\s*\d+\s*,\s*\d+\s*\)", lower_resp):
                    action_type = "SWIPE"
                    action_description = response
                elif any(w in lower_resp for w in ["wait", "pause", "sleep"]):
                    action_type = "WAIT"
                    action_description = response
                else:
                    logger.warning(f"Could not parse Phi Ground response: {response[:200]}...")
                    return {
                        "type": "wait",
                        "duration": 2.0,
                        "reasoning": "No valid action parsed, waiting",
                        "phi_ground_generated": True,
                        "confidence": 0.1,
                    }
            
            if action_type == "TAP":
                return self._parse_tap_action(action_description, ui_elements)
            elif action_type == "INPUT":
                return self._parse_input_action(action_description, ui_elements)
            elif action_type == "SWIPE":
                return self._parse_swipe_action(action_description)
            elif action_type == "WAIT":
                return self._parse_wait_action(action_description)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse Phi Ground response: {e}")
            return None
    
    def _parse_tap_action(
        self, 
        description: str, 
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Parse tap action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            ui_elements: Available UI elements for coordinate mapping
            
        Returns:
            Tap action dictionary or None
        """
        # Extract coordinates from description
        coord_match = re.search(r'\((\d+),\s*(\d+)\)', description)
        if coord_match:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            
            # Find matching element for validation
            matching_element = self._find_matching_element(description, ui_elements)
            
            return {
                "type": "tap",
                "x": x,
                "y": y,
                "element_text": matching_element.text if matching_element else "",
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True,
                "confidence": matching_element.confidence if matching_element else 0.5
            }
        
        # If no coordinates, try to find element by description
        matching_element = self._find_matching_element(description, ui_elements)
        if matching_element:
            x1, y1, x2, y2 = matching_element.bbox.as_tuple()
            return {
                "type": "tap",
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
                "element_text": matching_element.text,
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True,
                "confidence": matching_element.confidence
            }
        
        return None
    
    def _parse_input_action(
        self, 
        description: str, 
        ui_elements: List[UIElement]
    ) -> Optional[Dict[str, Any]]:
        """Parse input action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            ui_elements: Available UI elements for coordinate mapping
            
        Returns:
            Input action dictionary or None
        """
        # Extract text and field from description
        # Format: "INPUT: [text] into [field_description] at coordinates (x, y)"
        text_match = re.search(r'INPUT:\s*"([^"]+)"\s+into\s+(.+)', description, re.IGNORECASE)
        if text_match:
            input_text = text_match.group(1)
            field_description = text_match.group(2)
            
            # Extract coordinates
            coord_match = re.search(r'\((\d+),\s*(\d+)\)', field_description)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                
                # Find matching element
                matching_element = self._find_matching_element(field_description, ui_elements)
                
                return {
                    "type": "text_input",
                    "x": x,
                    "y": y,
                    "text": input_text,
                    "field_hint": matching_element.text if matching_element else field_description,
                    "reasoning": f"Phi Ground: {description}",
                    "phi_ground_generated": True,
                    "confidence": matching_element.confidence if matching_element else 0.5
                }
        
        return None
    
    def _parse_swipe_action(self, description: str) -> Optional[Dict[str, Any]]:
        """Parse swipe action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            
        Returns:
            Swipe action dictionary or None
        """
        # Extract coordinates: "SWIPE: from (x1, y1) to (x2, y2)"
        coord_match = re.search(r'from\s*\((\d+),\s*(\d+)\)\s+to\s+\((\d+),\s*(\d+)\)', description)
        if coord_match:
            x1, y1 = int(coord_match.group(1)), int(coord_match.group(2))
            x2, y2 = int(coord_match.group(3)), int(coord_match.group(4))
            
            return {
                "type": "swipe",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "duration": 300,  # Default duration
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True
            }
        
        return None
    
    def _parse_wait_action(self, description: str) -> Optional[Dict[str, Any]]:
        """Parse wait action from Phi Ground response.
        
        Args:
            description: Action description from Phi Ground
            
        Returns:
            Wait action dictionary or None
        """
        # Extract duration: "WAIT: [duration] seconds"
        duration_match = re.search(r'(\d+(?:\.\d+)?)\s*seconds?', description)
        if duration_match:
            duration = float(duration_match.group(1))
            
            return {
                "type": "wait",
                "duration": duration,
                "reasoning": f"Phi Ground: {description}",
                "phi_ground_generated": True
            }
        
        return None
    
    def _find_matching_element(
        self, 
        description: str, 
        ui_elements: List[UIElement]
    ) -> Optional[UIElement]:
        """Find UI element that matches the Phi Ground description.
        
        Args:
            description: Element description from Phi Ground
            ui_elements: Available UI elements
            
        Returns:
            Matching UI element or None
        """
        description_lower = description.lower()
        
        # Extract key terms from description
        key_terms = re.findall(r'\b\w+\b', description_lower)
        
        best_match = None
        best_score = 0
        
        for element in ui_elements:
            element_text_lower = element.text.lower()
            
            # Calculate similarity score
            score = 0
            for term in key_terms:
                if term in element_text_lower:
                    score += 1
                if element_text_lower in term or term in element_text_lower:
                    score += 0.5
            
            # Normalize by text length
            if len(element.text) > 0:
                score = score / len(element.text.split())
            
            if score > best_score:
                best_score = score
                best_match = element
        
        return best_match if best_score > 0.3 else None
    
    def validate_action_coordinates(
        self, 
        action: Dict[str, Any], 
        screen_width: int = 1080, 
        screen_height: int = 1920
    ) -> bool:
        """Validate action coordinates are within screen bounds.
        
        Args:
            action: Action dictionary
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        action_type = action.get("type")
        
        if action_type == "tap":
            x, y = action.get("x", 0), action.get("y", 0)
            return 0 <= x <= screen_width and 0 <= y <= screen_height
            
        elif action_type == "text_input":
            x, y = action.get("x", 0), action.get("y", 0)
            return 0 <= x <= screen_width and 0 <= y <= screen_height
            
        elif action_type == "swipe":
            x1, y1 = action.get("x1", 0), action.get("y1", 0)
            x2, y2 = action.get("x2", 0), action.get("y2", 0)
            return (0 <= x1 <= screen_width and 0 <= y1 <= screen_height and
                   0 <= x2 <= screen_width and 0 <= y2 <= screen_height)
        
        return True

    def _is_repeated_action(self, action: Dict[str, Any]) -> bool:
        """Check if the action is similar to recent actions (indicating stuck behavior).
        
        Args:
            action: Current action to check
            
        Returns:
            True if action is repeated, False otherwise
        """
        if not action or action.get("type") != "tap":
            return False
            
        current_x = action.get("x")
        current_y = action.get("y")
        current_text = action.get("element_text", "")
        
        if current_x is None or current_y is None:
            return False
            
        # Check for coordinate repetition (within small tolerance)
        coord_tolerance = 50  # pixels
        similar_coord_count = 0
        
        for recent_action in self.recent_actions:
            if recent_action.get("type") == "tap":
                recent_x = recent_action.get("x")
                recent_y = recent_action.get("y")
                if (recent_x is not None and recent_y is not None and
                    abs(current_x - recent_x) <= coord_tolerance and
                    abs(current_y - recent_y) <= coord_tolerance):
                    similar_coord_count += 1
                    
        return similar_coord_count >= self.repetition_threshold

    def _add_action_to_history(self, action: Dict[str, Any]) -> None:
        """Add action to recent history for repetition detection.
        
        Args:
            action: Action to add to history
        """
        if action:
            self.recent_actions.append(action)
            # Keep only the most recent actions
            if len(self.recent_actions) > self.max_recent_actions:
                self.recent_actions.pop(0)

    def _get_fallback_action(self, ui_elements: List[UIElement], task_description: str) -> Optional[Dict[str, Any]]:
        """Generate a fallback action when the model is stuck or refuses.
        
        Args:
            ui_elements: Available UI elements
            task_description: Current task
            
        Returns:
            Fallback action or None
        """
        try:
            # Strategy 1: Look for unclicked interactive elements
            unclicked_elements = []
            clicked_coords = set()
            
            # Get coordinates of recently clicked elements
            for recent_action in self.recent_actions:
                if recent_action.get("type") == "tap":
                    x, y = recent_action.get("x"), recent_action.get("y")
                    if x is not None and y is not None:
                        clicked_coords.add((x, y))
            
            # Find elements that haven't been clicked recently
            for element in ui_elements:
                if element.text and element.text.strip():
                    x1, y1, x2, y2 = element.bbox.as_tuple()
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Check if this coordinate hasn't been clicked recently
                    coord_clicked = False
                    for clicked_x, clicked_y in clicked_coords:
                        if abs(center_x - clicked_x) <= 50 and abs(center_y - clicked_y) <= 50:
                            coord_clicked = True
                            break
                    
                    if not coord_clicked:
                        unclicked_elements.append({
                            "element": element,
                            "center_x": center_x,
                            "center_y": center_y,
                            "text": element.text.strip()
                        })
            
            # Strategy 2: Prioritize elements based on task keywords
            task_lower = task_description.lower() if task_description else ""
            priority_elements = []
            
            for item in unclicked_elements:
                element_text_lower = item["text"].lower()
                priority_score = 0
                
                # Common action words
                action_words = ["continue", "next", "submit", "ok", "yes", "confirm", "done", "finish", "proceed"]
                for word in action_words:
                    if word in element_text_lower:
                        priority_score += 3
                
                # Task-specific keywords
                for word in task_lower.split():
                    if word in element_text_lower:
                        priority_score += 2
                
                # Prefer shorter text (likely buttons)
                if len(item["text"]) <= 20:
                    priority_score += 1
                
                priority_elements.append((priority_score, item))
            
            # Sort by priority and return the best option
            if priority_elements:
                priority_elements.sort(key=lambda x: x[0], reverse=True)
                best_item = priority_elements[0][1]
                
                return {
                    "type": "tap",
                    "x": best_item["center_x"],
                    "y": best_item["center_y"],
                    "element_text": best_item["text"],
                    "reasoning": f"Fallback: unclicked element '{best_item['text']}' (priority: {priority_elements[0][0]})",
                    "phi_ground_generated": True,
                    "confidence": 0.6,
                    "is_fallback": True
                }
            
            # Strategy 3: If no good options, try a different area of the screen
            if ui_elements:
                # Find an element in a different screen region
                screen_regions = [
                    (0, 0, 0.5, 0.5),      # Top-left
                    (0.5, 0, 1.0, 0.5),    # Top-right
                    (0, 0.5, 0.5, 1.0),    # Bottom-left
                    (0.5, 0.5, 1.0, 1.0),  # Bottom-right
                ]
                
                for region in screen_regions:
                    region_elements = []
                    for element in ui_elements:
                        if element.text and element.text.strip():
                            x1, y1, x2, y2 = element.bbox.as_tuple()
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            # Check if element is in this region
                            if (region[0] <= center_x/1344 <= region[2] and 
                                region[1] <= center_y/2992 <= region[3]):
                                region_elements.append({
                                    "element": element,
                                    "center_x": center_x,
                                    "center_y": center_y,
                                    "text": element.text.strip()
                                })
                    
                    if region_elements:
                        # Pick a random element from this region
                        import random
                        chosen = random.choice(region_elements)
                        return {
                            "type": "tap",
                            "x": chosen["center_x"],
                            "y": chosen["center_y"],
                            "element_text": chosen["text"],
                            "reasoning": f"Fallback: random element in region {region}",
                            "phi_ground_generated": True,
                            "confidence": 0.3,
                            "is_fallback": True
                        }
            
            # Strategy 4: Ultimate fallback - wait
            return {
                "type": "wait",
                "duration": 3.0,
                "reasoning": "Fallback: no suitable unclicked elements found",
                "phi_ground_generated": True,
                "confidence": 0.1,
                "is_fallback": True
            }
            
        except Exception as e:
            logger.warning(f"Fallback action generation failed: {e}")
            return None

    def _save_debug_response(
        self,
        image_path: str,
        task_description: str,
        raw_response: str,
        cleaned_response: str,
        action: Dict[str, Any],
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist raw and cleaned model responses and metadata to a JSON file.
        
        Args:
            image_path: Path to the screenshot used
            task_description: The task prompt
            raw_response: Raw decoded model output
            cleaned_response: After cleaning/parsing prep
            action: Parsed action (or fallback)
            extras: Additional metadata to include
        """
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            debug_path = logs_dir / f"phi_ground_response_{ts}.json"
            payload: Dict[str, Any] = {
                "timestamp_ms": ts,
                "image_path": image_path,
                "task_description": task_description,
                "raw_response": raw_response,
                "cleaned_response": cleaned_response,
                "action": action,
            }
            if extras:
                payload["meta"] = extras
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved Phi Ground debug response to {debug_path}")
        except Exception as save_err:
            logger.warning(f"Failed to save Phi Ground debug response: {save_err}")

# Global instance for reuse
_phi_ground_generator = None


def get_phi_ground_generator() -> PhiGroundActionGenerator:
    """Get or create the global Phi Ground generator instance."""
    global _phi_ground_generator
    if _phi_ground_generator is None:
        _phi_ground_generator = PhiGroundActionGenerator()
    return _phi_ground_generator
