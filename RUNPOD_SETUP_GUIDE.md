# 🚀 RunPod Distributed Setup Guide

This guide will help you set up AA_VA-Phi on RunPod to work with your local Android emulator.

## 📋 Prerequisites

### Local Machine (Your Laptop)
- ✅ Android emulator running
- ✅ ADB connected to emulator
- ✅ Local IP address (use `ifconfig` to find it)
- ✅ Port 8000 available

### RunPod Instance
- ✅ Jupyter Lab environment
- ✅ Python 3.8+ with pip
- ✅ Internet access

## 🔧 Step 1: Upload Code to RunPod

### Option A: Git Clone (Recommended)
```bash
# On RunPod, clone your repository
git clone https://github.com/MohnishBangaru/AA_VA-Phi.git
cd AA_VA-Phi
```

### Option B: Manual Upload
1. **Zip your local repository**:
   ```bash
   # On your local machine
   cd /Users/mohnishbangaru/Drizz/AA_VA-Phi
   zip -r AA_VA-Phi.zip . -x "*.pyc" "__pycache__/*" "*.log" "test_reports/*"
   ```

2. **Upload to RunPod**:
   - Go to RunPod Jupyter Lab
   - Navigate to `/workspace/`
   - Upload `AA_VA-Phi.zip`
   - Extract the zip file

## 🔧 Step 2: Install Dependencies on RunPod

```bash
# On RunPod, install required packages
cd /workspace/AA_VA-Phi

# Option A: Install from requirements.txt
pip install -r requirements.txt

# Option B: Use the dependency fixer (recommended for RunPod)
python scripts/fix_runpod_dependencies.py

# Option C: Emergency fix for critical dependencies
python scripts/emergency_fix_runpod.py

# Install additional dependencies for distributed setup
pip install fastapi uvicorn aiohttp requests
```

## 🔧 Step 3: Start Local ADB Server

### On Your Local Machine
```bash
# Start the local ADB server
cd /Users/mohnishbangaru/Drizz/AA_VA-Phi
python scripts/local_adb_server.py --host 0.0.0.0 --port 8000
```

**Keep this running!** This server bridges RunPod with your local emulator.

## 🔧 Step 4: Configure RunPod

### Create Minimal Configuration (Recommended)
On RunPod, create a minimal configuration that doesn't require API keys:

```bash
# On RunPod
cd /workspace/AA_VA-Phi
python scripts/runpod_minimal_config.py
```

### Create Custom Configuration (Optional)
If you want to customize settings, create `distributed_config.env`:

```bash
# On RunPod
cd /workspace/AA_VA-Phi
cat > distributed_config.env << EOF
# Local ADB Server Configuration
LOCAL_ADB_HOST=YOUR_LAPTOP_IP
LOCAL_ADB_PORT=8000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001

# WebSocket Configuration
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8002

# Testing Configuration
DEFAULT_ACTIONS=10
SCREENSHOT_DELAY=2
EOF
```

**Replace `YOUR_LAPTOP_IP` with your actual laptop IP address.**

## 🔧 Step 5: Test Connectivity

### Quick Test
```bash
# On RunPod
cd /workspace/AA_VA-Phi
python scripts/quick_start_distributed.py --local-ip YOUR_LAPTOP_IP
```

### Expected Output
```
✅ Connectivity Test: PASSED
✅ Device Operations Test: PASSED
✅ Screenshot Test: PASSED
```

## 🔧 Step 6: Run Distributed APK Test

### Upload APK to RunPod
1. Upload your APK file to `/workspace/AA_VA-Phi/`
2. Note the filename

### Run Test
```bash
# On RunPod
cd /workspace/AA_VA-Phi
python scripts/distributed_apk_tester.py \
    --apk com.Dominos_12.1.16-299_minAPI23\(arm64-v8a,armeabi-v7a,x86,x86_64\)\(nodpi\)_apkmirror.com.apk \
    --local-server http://YOUR_LAPTOP_IP:8000 \
    --actions 10
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "No such file or directory"
**Problem**: Files not found on RunPod
**Solution**: 
```bash
# Check if files exist
ls -la /workspace/AA_VA-Phi/scripts/
# If empty, re-upload the repository
```

#### 2. Connection Refused
**Problem**: Can't connect to local ADB server
**Solution**:
```bash
# Check if server is running locally
curl http://YOUR_LAPTOP_IP:8000/health
# If failed, restart local server
```

#### 3. Import Errors
**Problem**: Module not found (e.g., `ModuleNotFoundError: No module named 'dotenv'` or `pydantic_settings`)
**Solution**:
```bash
# Emergency fix (recommended)
python scripts/emergency_fix_runpod.py

# Or use the dependency fixer
python scripts/fix_runpod_dependencies.py

# Or install specific missing packages
pip install python-dotenv pydantic-settings fastapi uvicorn aiohttp requests

# Or install all requirements
pip install -r requirements.txt
```

#### 4. Configuration Errors
**Problem**: `ValidationError: openai_api_key Field required`
**Solution**:
```bash
# Create minimal configuration (recommended)
python scripts/runpod_minimal_config.py

# Or set up API keys interactively
python scripts/setup_api_keys.py

# Or manually create .env file
echo "OPENAI_API_KEY=" > .env
```

#### 4. ADB Device Not Found
**Problem**: No devices connected
**Solution**:
```bash
# On local machine, check ADB
adb devices
# Restart emulator if needed
```

## 📁 File Structure on RunPod

After setup, your RunPod should have:
```
/workspace/AA_VA-Phi/
├── scripts/
│   ├── distributed_apk_tester.py
│   ├── quick_start_distributed.py
│   └── local_adb_server.py
├── src/
│   ├── core/
│   ├── ai/
│   ├── vision/
│   └── ...
├── requirements.txt
├── distributed_config.env
└── [your-apk-file].apk
```

## 🎯 Quick Commands Reference

### Local Machine
```bash
# Start ADB server
python scripts/local_adb_server.py --host 0.0.0.0 --port 8000

# Check IP address
ifconfig | grep "inet " | grep -v 127.0.0.1
```

### RunPod
```bash
# Test connectivity
python scripts/quick_start_distributed.py --local-ip YOUR_IP

# Run APK test
python scripts/distributed_apk_tester.py --apk app.apk --local-server http://YOUR_IP:8000

# Check logs
tail -f distributed_test_reports/test.log
```

## 🚀 Production Setup

For production use:

1. **Use Environment Variables**:
   ```bash
   export LOCAL_ADB_HOST=YOUR_IP
   export LOCAL_ADB_PORT=8000
   ```

2. **Run in Background**:
   ```bash
   # Local: Run ADB server in background
   nohup python scripts/local_adb_server.py --host 0.0.0.0 --port 8000 &
   
   # RunPod: Run tests in background
   nohup python scripts/distributed_apk_tester.py --apk app.apk &
   ```

3. **Monitor Logs**:
   ```bash
   # Check server status
   curl http://YOUR_IP:8000/health
   
   # Monitor test progress
   tail -f distributed_test_reports/*.log
   ```

## 📞 Support

If you encounter issues:

1. Check the logs in `distributed_test_reports/`
2. Verify network connectivity
3. Ensure all dependencies are installed
4. Confirm ADB server is running locally

---

**Happy Testing! 🎉**
