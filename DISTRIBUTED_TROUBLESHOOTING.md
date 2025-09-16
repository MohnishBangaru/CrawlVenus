# Distributed Testing Troubleshooting Guide

This guide helps you resolve common issues with distributed testing between RunPod and your local emulator.

## 🔍 **Error: Cannot connect to local ADB server**

### **Problem**
```
ERROR - Request failed: 503 Server Error: Service Unavailable for url: https://dominos-test.loca.lt/health
ERROR - Failed to connect to local ADB server
```

### **Root Cause**
The local ADB server at `https://dominos-test.loca.lt` is not running or not accessible.

### **Solution Steps**

#### **Step 1: Check Local Setup**
On your local machine (laptop/desktop):

```bash
# Check if ADB is working
adb devices

# Check if local ADB server is running
curl http://localhost:8000/health
```

#### **Step 2: Start Local ADB Server**
If the local server is not running:

```bash
# Install dependencies (if not already installed)
pip install fastapi uvicorn

# Start the local ADB server
python scripts/local_adb_server.py --host 0.0.0.0 --port 8000
```

#### **Step 3: Setup Tunnel**
If you need a public URL for RunPod to access:

```bash
# Install Node.js (if not already installed)
# Download from: https://nodejs.org/

# Install localtunnel
npm install -g localtunnel

# Start tunnel
lt --port 8000 --subdomain dominos-test
```

#### **Step 4: Test Connection**
Use the setup script to verify everything is working:

```bash
# Check current setup
python scripts/setup_distributed_testing.py --mode check

# Start everything
python scripts/setup_distributed_testing.py --mode full --public-url dominos-test.loca.lt
```

## 🚀 **Quick Fix Commands**

### **Option 1: Use Setup Script**
```bash
python scripts/setup_distributed_testing.py --mode full --public-url dominos-test.loca.lt
```

### **Option 2: Manual Setup**
```bash
# Terminal 1: Start local ADB server
python scripts/local_adb_server.py --host 0.0.0.0 --port 8000

# Terminal 2: Start tunnel
lt --port 8000 --subdomain dominos-test

# Terminal 3: Test from RunPod
python scripts/distributed_apk_tester.py \
  --apk com.Dominos_12.1.16-299_minAPI23\(arm64-v8a,armeabi-v7a,x86,x86_64\)\(nodpi\)_apkmirror.com.apk \
  --local-server https://dominos-test.loca.lt \
  --actions 10
```

## 🔧 **Common Issues & Solutions**

### **Issue 1: ADB devices not found**
```bash
# Solution: Start Android emulator
# For Android Studio emulator:
# 1. Open Android Studio
# 2. Go to AVD Manager
# 3. Start your emulator

# For command line:
emulator -avd Pixel_4_API_30
```

### **Issue 2: Permission denied**
```bash
# Solution: Check ADB authorization
adb devices
# If device shows "unauthorized", check the emulator screen for authorization prompt
```

### **Issue 3: Port already in use**
```bash
# Solution: Use different port
python scripts/local_adb_server.py --host 0.0.0.0 --port 8001
# Then update tunnel: lt --port 8001 --subdomain dominos-test
```

### **Issue 4: Tunnel not working**
```bash
# Solution: Try different tunnel service
# Option A: ngrok
ngrok http 8000

# Option B: cloudflared
cloudflared tunnel --url http://localhost:8000
```

## 📋 **Checklist**

Before running distributed tests, ensure:

- [ ] Android emulator is running
- [ ] ADB can see the device (`adb devices`)
- [ ] Local ADB server is running (`curl http://localhost:8000/health`)
- [ ] Tunnel is active (if using public URL)
- [ ] RunPod can reach the tunnel URL

## 🆘 **Getting Help**

If you're still having issues:

1. **Check logs**: Look at the detailed error messages
2. **Verify network**: Ensure your local machine can accept incoming connections
3. **Try local testing**: Test without tunnel first using `localhost`
4. **Check firewall**: Ensure port 8000 is not blocked

## 📞 **Alternative Solutions**

### **Option A: Use ngrok instead of localtunnel**
```bash
# Install ngrok
# Download from: https://ngrok.com/

# Start tunnel
ngrok http 8000

# Use the provided HTTPS URL in your RunPod command
```

### **Option B: Use cloudflared tunnel**
```bash
# Install cloudflared
# Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/

# Start tunnel
cloudflared tunnel --url http://localhost:8000
```

### **Option C: Direct connection (same network)**
If RunPod and your local machine are on the same network:
```bash
# Use your local IP address
python scripts/distributed_apk_tester.py \
  --apk your_app.apk \
  --local-server http://192.168.1.100:8000
```
