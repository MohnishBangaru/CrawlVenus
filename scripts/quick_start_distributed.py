#!/usr/bin/env python3
"""
Quick Start Script for Distributed AA_VA-Phi Setup
==================================================

This script helps you quickly set up and test the distributed RunPod + Local Emulator configuration.

Usage:
    python scripts/quick_start_distributed.py --local-ip YOUR_LAPTOP_IP
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.distributed_device_manager import test_distributed_connection, DistributedDeviceManager
from src.core.distributed_config import distributed_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_connectivity(local_ip: str, port: int = 8000):
    """Test connectivity to local ADB server."""
    # Handle both full URLs and just hostnames
    if local_ip.startswith(('http://', 'https://')):
        url = local_ip
        logger.info(f"Testing connectivity to {url}")
    else:
        url = f"http://{local_ip}:{port}"
        logger.info(f"Testing connectivity to {url}")
    
    success = await test_distributed_connection(url)
    
    if success:
        logger.info("✅ Connectivity test passed!")
        return True
    else:
        logger.error("❌ Connectivity test failed!")
        return False


async def test_device_operations(local_ip: str, port: int = 8000):
    """Test basic device operations."""
    logger.info("Testing device operations...")
    
    # Handle both full URLs and just hostnames
    if local_ip.startswith(('http://', 'https://')):
        url = local_ip
    else:
        url = f"http://{local_ip}:{port}"
    
    try:
        async with DistributedDeviceManager(url) as manager:
            # Test device detection
            devices = manager.get_devices()
            logger.info(f"Found {len(devices)} devices")
            
            if not devices:
                logger.warning("No devices found. Make sure your emulator is running.")
                return False
            
            # Test screenshot
            screenshot_path = f"test_screenshot_{int(time.time())}.png"
            screenshot = manager.take_screenshot(screenshot_path)
            
            if screenshot:
                logger.info(f"✅ Screenshot test passed: {screenshot}")
                # Clean up test screenshot
                Path(screenshot_path).unlink(missing_ok=True)
            else:
                logger.error("❌ Screenshot test failed")
                return False
            
            # Test foreground app detection
            foreground = manager.get_foreground_app()
            if foreground:
                logger.info(f"✅ Foreground app detection: {foreground}")
            else:
                logger.warning("⚠️ Could not detect foreground app")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Device operations test failed: {e}")
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    # Check required packages
    required_packages = [
        'fastapi', 'uvicorn', 'aiohttp', 'requests',
        'PIL', 'pydantic', 'pydantic_settings'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ Prerequisites check passed!")
    return True


def generate_config_template(local_ip: str):
    """Generate configuration template."""
    logger.info("Generating configuration template...")
    
    config_content = f"""# Distributed AA_VA-Phi Configuration
# Copy this to your .env file

# Distributed Configuration
DISTRIBUTED_MODE=true
LOCAL_ADB_HOST={local_ip}
LOCAL_ADB_PORT=8000

# AI Configuration
OPENAI_API_KEY=your_openai_api_key_here
PHI_GROUND_MODEL=microsoft/Phi-3-vision-128k-instruct

# Network Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
"""
    
    config_file = Path("distributed_config.env")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info(f"✅ Configuration template saved to {config_file}")
    logger.info("Edit this file and copy to .env")


def print_setup_instructions(local_ip: str):
    """Print setup instructions."""
    logger.info("📋 Setup Instructions:")
    logger.info("=" * 50)
    
    print(f"""
🚀 DISTRIBUTED SETUP INSTRUCTIONS

1. ON YOUR LAPTOP:
   - Start your Android emulator
   - Run: python scripts/local_adb_server.py --host 0.0.0.0 --port 8000
   - Your laptop IP: {local_ip}

2. ON RUNPOD:
   - Clone: git clone https://github.com/MohnishBangaru/AA_VA-Phi.git
   - Install: pip install -r requirements.txt
   - Copy distributed_config.env to .env and edit
   - Run this script to test: python scripts/quick_start_distributed.py --local-ip {local_ip}

3. TEST CONNECTION:
   - This script will test connectivity and basic operations
   - If successful, you can run distributed tests

4. RUN DISTRIBUTED TEST:
   python scripts/distributed_apk_tester.py \\
       --apk /path/to/app.apk \\
       --local-server {local_ip} \\
       --actions 10
   
   Results will be saved to: /Users/mohnishbangaru/Drizz/local_test_reports/

📚 For detailed instructions, see: DISTRIBUTED_SETUP_GUIDE.md
""")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick Start for Distributed AA_VA-Phi")
    parser.add_argument("--local-ip", required=True, help="Your laptop's IP address")
    parser.add_argument("--port", type=int, default=8000, help="Local ADB server port")
    parser.add_argument("--skip-tests", action="store_true", help="Skip connectivity tests")
    
    args = parser.parse_args()
    
    logger.info("🚀 Quick Start for Distributed AA_VA-Phi Setup")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Generate config template
    generate_config_template(args.local_ip)
    
    # Print setup instructions
    print_setup_instructions(args.local_ip)
    
    if args.skip_tests:
        logger.info("⏭️ Skipping connectivity tests")
        return
    
    # Test connectivity
    logger.info("\n🔍 Testing connectivity...")
    if not await test_connectivity(args.local_ip, args.port):
        logger.error("❌ Setup incomplete. Please check your configuration.")
        sys.exit(1)
    
    # Test device operations
    logger.info("\n🔍 Testing device operations...")
    if not await test_device_operations(args.local_ip, args.port):
        logger.error("❌ Device operations failed. Please check your emulator.")
        sys.exit(1)
    
    logger.info("\n🎉 Setup completed successfully!")
    logger.info("You can now run distributed APK tests.")
    logger.info("See DISTRIBUTED_SETUP_GUIDE.md for detailed instructions.")


if __name__ == "__main__":
    asyncio.run(main())
