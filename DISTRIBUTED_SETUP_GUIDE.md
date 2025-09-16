# Distributed Setup Guide: RunPod + Local Emulator

This guide explains how to set up AA_VA-Phi to run on RunPod while your Android emulator runs on your local laptop.

## 🏗️ Architecture Overview

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   RunPod        │ ◄──────────────────► │   Your Laptop   │
│   (Cloud)       │                      │   (Local)       │
│                 │                      │                 │
│ • AI Processing │                      │ • Android       │
│ • Vision AI     │                      │   Emulator      │
│ • Phi Ground    │                      │ • ADB Server    │
│ • Test Logic    │                      │ • Device        │
│                 │                      │   Interaction   │
└─────────────────┘                      └─────────────────┘
```

## 📋 Prerequisites

### On Your Laptop (Local Setup)
- ✅ Android Studio with emulator
- ✅ ADB (Android Debug Bridge)
- ✅ Python 3.8+
- ✅ FastAPI and uvicorn
- ✅ Network connectivity (same network as RunPod)

### On RunPod
- ✅ Jupyter Lab environment
- ✅ AA_VA-Phi codebase
- ✅ GPU support (for Phi Ground)
- ✅ Network access to your laptop

## 🚀 Step-by-Step Setup

### Step 1: Local Laptop Setup

#### 1.1 Install Dependencies
```bash
# Install FastAPI and uvicorn
pip install fastapi uvicorn

# Verify ADB is available
adb version
```

#### 1.2 Start Android Emulator
```bash
# Start your Android emulator
# This can be done through Android Studio or command line
emulator -avd YOUR_AVD_NAME
```

#### 1.3 Verify ADB Connection
```bash
# Check if emulator is connected
adb devices

# Should show something like:
# emulator-5554    device
```

#### 1.4 Start Local ADB Server
```bash
# Navigate to your AA_VA-Phi directory
cd /path/to/AA_VA-Phi

# Start the local ADB server
python scripts/local_adb_server.py --host 0.0.0.0 --port 8000

# The server will be available at http://YOUR_LAPTOP_IP:8000
```

#### 1.5 Find Your Laptop's IP Address
```bash
# On macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# On Windows
ipconfig | findstr "IPv4"
```

### Step 2: RunPod Setup

#### 2.1 Clone/Upload AA_VA-Phi
```bash
# Clone the repository on RunPod
git clone https://github.com/MohnishBangaru/AA_VA-Phi.git
cd AA_VA-Phi

# Or upload the codebase through Jupyter Lab
```

#### 2.2 Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install additional distributed dependencies
pip install aiohttp requests fastapi uvicorn
```

#### 2.3 Configure Environment
```bash
# Copy environment template
cp env.example .env

# Edit .env file with your configuration
nano .env
```

Add these variables to your `.env`:
```env
# Distributed Configuration
DISTRIBUTED_MODE=true
LOCAL_ADB_HOST=YOUR_LAPTOP_IP
LOCAL_ADB_PORT=8000

# AI Configuration
OPENAI_API_KEY=your_openai_api_key
PHI_GROUND_MODEL=microsoft/Phi-3-vision-128k-instruct

# Network Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

#### 2.4 Test Connection
```python
# In Jupyter Lab, test the connection
import asyncio
from src.core.distributed_device_manager import test_distributed_connection

# Test connection to your laptop
success = await test_distributed_connection("http://YOUR_LAPTOP_IP:8000")
print(f"Connection successful: {success}")
```

### Step 3: Run Distributed Test

#### 3.1 Upload APK to RunPod
```bash
# Upload your APK file to RunPod
# You can do this through Jupyter Lab file upload or SCP
```

#### 3.2 Run the Distributed Test
```bash
# Run the distributed APK tester
python scripts/distributed_apk_tester.py \
    --apk /path/to/your/app.apk \
    --local-server http://YOUR_LAPTOP_IP:8000 \
    --actions 15 \
    --output-dir test_reports
```

#### 3.3 Monitor Progress
The test will:
1. Connect to your local ADB server
2. Install the APK on your emulator
3. Take screenshots via the local server
4. Process images with AI on RunPod
5. Send actions back to your emulator
6. Generate comprehensive reports

## 🔧 Configuration Options

### Local ADB Server Configuration
```python
# scripts/local_adb_server.py
--host 0.0.0.0          # Bind to all interfaces
--port 8000             # Port for the server
--debug                 # Enable debug mode
```

### Distributed Configuration
```python
# src/core/distributed_config.py
distributed_mode: bool = True
local_adb_host: str = "YOUR_LAPTOP_IP"
local_adb_port: int = 8000
connection_timeout: int = 30
```

## 🌐 Network Configuration

### Firewall Settings
Make sure port 8000 is open on your laptop:
```bash
# macOS
sudo pfctl -e
echo "pass in proto tcp from any to any port 8000" | sudo pfctl -f -

# Linux
sudo ufw allow 8000

# Windows
# Configure Windows Firewall to allow port 8000
```

### VPN Considerations
If using VPN:
- Ensure both devices are on the same network
- Use local IP addresses, not public ones
- Consider port forwarding if needed

## 📊 Monitoring and Debugging

### Local Server Logs
```bash
# Monitor local ADB server logs
tail -f local_adb_server.log
```

### RunPod Logs
```python
# In Jupyter Lab, check logs
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Network Connectivity Test
```python
# Test network connectivity
import requests
response = requests.get("http://YOUR_LAPTOP_IP:8000/health")
print(response.json())
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Connection Refused
```
Error: Connection refused to http://YOUR_LAPTOP_IP:8000
```
**Solution:**
- Check if local ADB server is running
- Verify IP address is correct
- Check firewall settings

#### 2. ADB Device Not Found
```
Error: No devices found
```
**Solution:**
- Ensure emulator is running
- Check `adb devices` on laptop
- Restart ADB server

#### 3. Screenshot Failures
```
Error: Failed to take screenshot
```
**Solution:**
- Check emulator is responsive
- Verify ADB connection
- Check disk space on laptop

#### 4. AI Model Loading Issues
```
Error: Phi Ground model failed to load
```
**Solution:**
- Check GPU availability on RunPod
- Verify model download
- Check memory usage

### Performance Optimization

#### 1. Reduce Latency
- Use wired network connection
- Minimize network hops
- Optimize screenshot quality

#### 2. Improve Throughput
- Use multiple ADB connections
- Batch operations
- Compress image transfers

#### 3. Memory Management
- Monitor GPU memory usage
- Clear cache between tests
- Use smaller model variants

## 📈 Advanced Features

### Custom Actions
```python
# Define custom actions
custom_actions = [
    {"type": "tap", "x": 500, "y": 800},
    {"type": "swipe", "start_x": 100, "start_y": 500, "end_x": 700, "end_y": 500},
    {"type": "keyevent", "key_code": 4}
]
```

### Batch Testing
```python
# Test multiple APKs
apks = ["app1.apk", "app2.apk", "app3.apk"]
for apk in apks:
    await run_distributed_test(apk, local_server_url)
```

### Real-time Monitoring
```python
# Monitor test progress in real-time
import asyncio
from src.core.distributed_device_manager import DistributedDeviceManager

async def monitor_test():
    async with DistributedDeviceManager(local_server_url) as manager:
        while True:
            devices = manager.get_devices()
            print(f"Connected devices: {len(devices)}")
            await asyncio.sleep(5)
```

## 🔒 Security Considerations

### Network Security
- Use HTTPS for production
- Implement authentication
- Restrict network access

### Data Privacy
- Don't store sensitive screenshots
- Clear test data after runs
- Use secure file transfers

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ADB Command Reference](https://developer.android.com/studio/command-line/adb)
- [RunPod Documentation](https://docs.runpod.io/)
- [Phi Ground Model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)

## 🎯 Next Steps

1. **Test with Simple APK**: Start with a basic app to verify setup
2. **Optimize Performance**: Tune network and AI parameters
3. **Scale Up**: Test with multiple devices or longer sessions
4. **Production Deployment**: Add monitoring and error handling

This distributed setup allows you to leverage RunPod's powerful GPU resources for AI processing while keeping your Android emulator running locally for better performance and control.
