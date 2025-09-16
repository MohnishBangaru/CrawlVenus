# AA_VA Testing Guide

This guide walks you through setting up the environment and executing the **Universal APK Tester** included in the AA_VA project so you can reproduce automated test runs and generate reports.

---

## 1. Prerequisites

1. **Operating System**: macOS (10.15 Catalina or newer recommended) – these steps can be adapted for Linux.
2. **Homebrew** (macOS only):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. **Python 3.10 or newer** (ensure it is the default `python3` on PATH).
4. **ADB** – part of Android SDK Platform Tools.
   ```bash
   brew install --cask android-platform-tools   # macOS
   # or download from https://developer.android.com/studio/releases/platform-tools
   ```
5. **Android device / emulator**:
   * Enable USB-debugging (device) **or** start an emulator in Android Studio.
   * Confirm connectivity:
     ```bash
     adb devices
     ```
     You should see at least one device listed as *device*.
6. **Tesseract-OCR** (required for on-screen text detection):
   ```bash
   brew install tesseract            # macOS
   sudo apt-get install tesseract-ocr  # Debian/Ubuntu
   ```
   Optional – install extra language packs if your app uses non-English UI.
7. **Git** (for cloning / updates) and **Make** (optional convenience).

---

## 2. Clone & Install Python Dependencies

```bash
# Clone the repo (or use your existing checkout)
git clone git@github.com:<YOUR_NAME>/AA_VA.git
cd AA_VA

# Create & activate a virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install all Python requirements
pip install -r requirements.txt
```

> **Tip :** If you use Apple Silicon you may need Rosetta for `opencv-python`; run `arch -x86_64 pip install opencv-python` in that case.

---

## 3. Environment Variables

Create a copy of `env.example` and edit as needed:

```bash
cp env.example .env
```

Key variables:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI key for LLM analysis |
| `USE_OCR`        | Should remain `true` so VisionEngine performs OCR |
| `TESSERACT_CMD`  | (Optional) Path to `tesseract` binary if not on PATH |
| `ANDROID_DEVICE_ID` | e.g. `emulator-5554` or the device’s serial |

Load env vars in the current shell:

```bash
export $(grep -v '^#' .env | xargs)
```

---

## 4. Verify Setup

1. **Tesseract**
   ```bash
   tesseract -v         # should print a version number
   ```
2. **ADB + device**
   ```bash
   adb devices          # device/emulator must be “device” not “offline”
   ```
3. **Python packages** – `python - <<<'import cv2, pytesseract; print("OK")'` should print `OK` without errors.

---

## 5. Running the Universal APK Tester

The key script lives at `scripts/universal_apk_tester.py`.

### Basic Usage

```bash
python scripts/universal_apk_tester.py \
  --apk /absolute/path/to/my_app.apk \
  --actions 25              # number of UI actions (default 10)
```

### Important Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--apk` | **Required** – path to the APK you want to test | `/Users/me/apps/example.apk` |
| `--actions` | How many UI actions to attempt | `30` |
| `--output_dir` | Where to store results | `my_test_output` |
| `--screenshot_frequency` | `minimal`, `adaptive`, or `high` | `adaptive` |
| `--max_screenshots_per_action` | Only used with `adaptive` | `3` |
| `--no-report` | Skip HTML/PDF report generation |  |

Example:

```bash
python scripts/universal_apk_tester.py \
  --apk ~/Downloads/com.example.shopping.apk \
  --actions 40 \
  --output_dir test_com_example \
  --screenshot_frequency adaptive \
  --max_screenshots_per_action 4
```

---

## 6. Output Structure

The tester creates an output directory (default `test_com_<appname>/`) containing:

```
test_com_<appname>/
  ├── analysis_data.json      # structured data for every state/action
  ├── screenshots/            # raw screenshots with timestamps
  └── final_report/
       ├── final_comprehensive_report.html
       ├── final_comprehensive_report.pdf
       ├── images/            # inline screenshot assets
       ├── graphs/            # UTG diagrams etc.
       └── ocr_images/        # OCR debug overlays
```

---