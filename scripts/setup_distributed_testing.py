#!/usr/bin/env python3
"""
Distributed Testing Setup Script
================================

This script helps set up and troubleshoot distributed testing between RunPod and local emulator.
"""

import argparse
import subprocess
import sys
import time
import requests
from pathlib import Path

def check_local_adb_server(url: str) -> bool:
    """Check if local ADB server is running."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Local ADB server not accessible: {e}")
        return False

def check_adb_devices() -> bool:
    """Check if ADB devices are available locally."""
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            devices = [line for line in lines if line.strip() and 'device' in line]
            if devices:
                print(f"✅ Found {len(devices)} ADB device(s):")
                for device in devices:
                    print(f"   {device}")
                return True
            else:
                print("❌ No ADB devices found")
                return False
        else:
            print(f"❌ ADB command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ ADB not found in PATH")
        return False

def start_local_adb_server(host: str = "0.0.0.0", port: int = 8000) -> bool:
    """Start the local ADB server."""
    script_path = Path(__file__).parent / "local_adb_server.py"
    
    if not script_path.exists():
        print(f"❌ Local ADB server script not found: {script_path}")
        return False
    
    print(f"🚀 Starting local ADB server on {host}:{port}...")
    
    try:
        # Start the server in background
        process = subprocess.Popen([
            sys.executable, str(script_path),
            "--host", host,
            "--port", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ Local ADB server started successfully")
            print(f"   URL: http://{host}:{port}")
            print(f"   Health check: http://{host}:{port}/health")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start local ADB server:")
            print(f"   STDOUT: {stdout.decode()}")
            print(f"   STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting local ADB server: {e}")
        return False

def test_connection(url: str) -> bool:
    """Test connection to local ADB server."""
    print(f"🔍 Testing connection to {url}...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ Health check passed: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test devices endpoint
    try:
        response = requests.get(f"{url}/devices", timeout=10)
        if response.status_code == 200:
            devices = response.json()
            print(f"✅ Devices endpoint working: {len(devices)} devices found")
        else:
            print(f"❌ Devices endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Devices endpoint failed: {e}")
        return False
    
    return True

def setup_tunnel(public_url: str, local_port: int = 8000) -> bool:
    """Setup tunnel using localtunnel or similar."""
    print(f"🌐 Setting up tunnel from {public_url} to localhost:{local_port}...")
    
    # Check if localtunnel is available
    try:
        result = subprocess.run(['npx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Node.js/npx available")
        else:
            print("❌ Node.js/npx not available")
            return False
    except FileNotFoundError:
        print("❌ Node.js/npx not found")
        return False
    
    # Start localtunnel
    try:
        print(f"🚀 Starting localtunnel...")
        process = subprocess.Popen([
            'npx', 'localtunnel', '--port', str(local_port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for tunnel to establish
        time.sleep(5)
        
        if process.poll() is None:
            print(f"✅ Tunnel started successfully")
            print(f"   Public URL: {public_url}")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start tunnel:")
            print(f"   STDOUT: {stdout.decode()}")
            print(f"   STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting tunnel: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup distributed testing")
    parser.add_argument("--mode", choices=["check", "start", "tunnel", "full"], 
                       default="check", help="Setup mode")
    parser.add_argument("--host", default="0.0.0.0", help="Local server host")
    parser.add_argument("--port", type=int, default=8000, help="Local server port")
    parser.add_argument("--public-url", help="Public URL for tunnel (e.g., dominos-test.loca.lt)")
    
    args = parser.parse_args()
    
    print("🔧 Distributed Testing Setup")
    print("=" * 40)
    
    if args.mode in ["check", "full"]:
        print("\n📋 Checking prerequisites...")
        
        # Check ADB devices
        if not check_adb_devices():
            print("\n💡 To fix ADB devices:")
            print("   1. Start your Android emulator")
            print("   2. Run: adb devices")
            print("   3. Make sure device shows as 'device' (not 'unauthorized')")
            return
        
        # Check local ADB server
        local_url = f"http://{args.host}:{args.port}"
        if check_local_adb_server(local_url):
            print("✅ Local ADB server is running")
        else:
            print("❌ Local ADB server is not running")
    
    if args.mode in ["start", "full"]:
        print("\n🚀 Starting local ADB server...")
        if not start_local_adb_server(args.host, args.port):
            print("\n💡 To fix local ADB server:")
            print("   1. Install dependencies: pip install fastapi uvicorn")
            print("   2. Make sure ADB is in your PATH")
            print("   3. Run: python scripts/local_adb_server.py --host 0.0.0.0 --port 8000")
            return
    
    if args.mode in ["tunnel", "full"]:
        if not args.public_url:
            print("❌ --public-url is required for tunnel mode")
            return
        
        print(f"\n🌐 Setting up tunnel to {args.public_url}...")
        if not setup_tunnel(args.public_url, args.port):
            print("\n💡 To fix tunnel:")
            print("   1. Install Node.js")
            print("   2. Run: npx localtunnel --port 8000")
            print("   3. Use the provided public URL")
            return
    
    if args.mode in ["check", "full"]:
        print("\n🔍 Testing connection...")
        if args.public_url:
            public_url = f"https://{args.public_url}"
            if test_connection(public_url):
                print("✅ All tests passed! Ready for distributed testing.")
            else:
                print("❌ Connection test failed")
        else:
            local_url = f"http://{args.host}:{args.port}"
            if test_connection(local_url):
                print("✅ Local connection working! Ready for local testing.")
            else:
                print("❌ Local connection test failed")
    
    print("\n📝 Next steps:")
    print("   1. Make sure your local ADB server is running")
    print("   2. If using tunnel, make sure it's active")
    print("   3. Run the distributed test from RunPod:")
    print(f"      python scripts/distributed_apk_tester.py --apk your_app.apk --local-server https://{args.public_url or 'localhost:8000'}")

if __name__ == "__main__":
    main()
