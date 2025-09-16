#!/usr/bin/env python3
"""Test script for Phi Ground integration.

This script tests the Phi Ground integration with AA_VA to ensure it works correctly
for generating touch actions instead of mouse actions.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.ai.phi_ground import get_phi_ground_generator
from src.vision.models import UIElement, BoundingBox
from src.core.config import config


async def test_phi_ground_integration():
    """Test Phi Ground integration with a sample screenshot."""
    
    logger.info("Testing Phi Ground integration...")
    
    # Check if Phi Ground is enabled
    if not config.use_phi_ground:
        logger.warning("Phi Ground is disabled in configuration")
        return
    
    try:
        # Initialize Phi Ground generator
        phi_ground = get_phi_ground_generator()
        await phi_ground.initialize()
        
        logger.info("✓ Phi Ground initialized successfully")
        
        # Create sample UI elements for testing
        sample_elements = [
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
        
        # Sample task and action history
        task_description = "Login to the application"
        action_history = [
            {"type": "tap", "element_text": "Login", "x": 200, "y": 225}
        ]
        
        # Test with a sample screenshot (if available)
        screenshot_path = "test_screenshot.png"
        if os.path.exists(screenshot_path):
            logger.info(f"Testing with screenshot: {screenshot_path}")
            
            # Generate action using Phi Ground
            action = await phi_ground.generate_touch_action(
                screenshot_path, task_description, action_history, sample_elements
            )
            
            if action:
                logger.info("✓ Phi Ground generated action successfully")
                logger.info(f"Action type: {action.get('type')}")
                logger.info(f"Reasoning: {action.get('reasoning')}")
                logger.info(f"Confidence: {action.get('confidence', 0.5):.2f}")
                
                # Validate coordinates
                if phi_ground.validate_action_coordinates(action):
                    logger.info("✓ Action coordinates are valid")
                else:
                    logger.warning("⚠ Action coordinates are invalid")
            else:
                logger.warning("⚠ Phi Ground did not generate an action")
        else:
            logger.info("No test screenshot found, testing without image...")
            
            # Test without screenshot (should fall back gracefully)
            action = await phi_ground.generate_touch_action(
                "", task_description, action_history, sample_elements
            )
            
            if action is None:
                logger.info("✓ Phi Ground correctly returned None when no screenshot available")
            else:
                logger.warning("⚠ Phi Ground generated action without screenshot (unexpected)")
        
        logger.info("✓ Phi Ground integration test completed")
        
    except Exception as e:
        logger.error(f"❌ Phi Ground integration test failed: {e}")
        raise


async def test_action_prioritizer_integration():
    """Test Phi Ground integration with action prioritizer."""
    
    logger.info("Testing Phi Ground integration with action prioritizer...")
    
    try:
        from src.core.action_prioritizer import get_action_prioritizer
        
        prioritizer = get_action_prioritizer()
        
        # Sample UI elements in dictionary format
        ui_elements = [
            {
                "text": "Login",
                "element_type": "button",
                "confidence": 0.9,
                "bounds": {"x": 100, "y": 200, "x2": 300, "y2": 250}
            },
            {
                "text": "Email",
                "element_type": "input",
                "confidence": 0.8,
                "bounds": {"x": 100, "y": 300, "x2": 400, "y2": 350}
            }
        ]
        
        task_description = "Login to the application"
        action_history = []
        
        # Test with screenshot path
        screenshot_path = "test_screenshot.png"
        if os.path.exists(screenshot_path):
            logger.info(f"Testing action prioritizer with screenshot: {screenshot_path}")
            
            optimal_action = prioritizer.get_optimal_action(
                ui_elements=ui_elements,
                task_description=task_description,
                action_history=action_history,
                screenshot_path=screenshot_path
            )
            
            if optimal_action:
                logger.info("✓ Action prioritizer with Phi Ground worked")
                logger.info(f"Action type: {optimal_action.action_type.name}")
                logger.info(f"Score: {optimal_action.score:.2f}")
                logger.info(f"Reasoning: {optimal_action.reasoning}")
            else:
                logger.warning("⚠ Action prioritizer did not find optimal action")
        else:
            logger.info("No test screenshot found, testing without Phi Ground...")
            
            optimal_action = prioritizer.get_optimal_action(
                ui_elements=ui_elements,
                task_description=task_description,
                action_history=action_history
            )
            
            if optimal_action:
                logger.info("✓ Action prioritizer worked without Phi Ground")
            else:
                logger.warning("⚠ Action prioritizer did not find optimal action")
        
        logger.info("✓ Action prioritizer integration test completed")
        
    except Exception as e:
        logger.error(f"❌ Action prioritizer integration test failed: {e}")
        raise


async def main():
    """Main test function."""
    logger.info("Starting Phi Ground integration tests...")
    
    # Test basic Phi Ground functionality
    await test_phi_ground_integration()
    
    # Test integration with action prioritizer
    await test_action_prioritizer_integration()
    
    logger.info("All Phi Ground integration tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
