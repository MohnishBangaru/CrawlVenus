# AA_VA – Autonomous Android Visual Analyzer

AA_VA is an AI-powered framework that installs any Android APK on an emulator/device, explores its UI with GPT-driven reasoning and computer-vision cues, and produces concise, professional PDF test reports.

## ✨ Key Features

• GPT-4/3.5 prompts decide the next UI action, emulating real user flows.  
• CV-based detector (`src/vision/`) finds clickable areas and prevents dead-ends.  
• Robust recovery modules keep the target app in the foreground and retry on crashes.  
• Screenshot & log capture for every step.  
• Polished PDF reports with graphs, OCR images, and action timeline.  
• One-shot runner script: `scripts/universal_apk_tester.py --apk /path/to/app.apk`.

## 🚀 Quick Start

```bash
# Clone and enter the project
$ git clone https://github.com/<your-username>/AA_VA.git
$ cd AA_VA

# Create Python 3.11 virtual environment (optional but recommended)
$ python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
$ pip install --upgrade pip
$ pip install -r requirements.txt

# Configure secrets (OpenAI key, etc.)
$ cp env.example .env
# ↳ edit .env and fill in the missing values

# Run a full exploratory test
$ python scripts/universal_apk_tester.py --apk /absolute/path/to/app.apk
```

Generated artifacts live under `test_<package_name>/`, including:
* `analysis_data.json` – structured log of every action/state.  
* `screenshots/` – per-action PNGs.  
* `final_report/…/final_comprehensive_report.pdf` – shareable test report.

## 📂 Project Structure (abridged)

```
AA_VA/
├── scripts/                # CLI entry points
├── src/
│   ├── ai/                 # Prompt building & GPT client
│   ├── automation/         # Action execution & planning
│   ├── core/               # Orchestration, recovery, logging
│   ├── utils/              # Helpers
│   └── vision/             # Screenshot capture & CV models
└── test_<app>/             # Per-APK run outputs
```