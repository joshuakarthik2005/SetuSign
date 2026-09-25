# SetuSign — Environment Setup

## Prerequisites

- **OS**: Windows 11 (ARM64 for NPU; x64 for development with CPU fallback)
- **Python**: 3.12.x (required — MediaPipe does not support 3.13+)
- **Git**: 2.x+

---

## Python Version

**MediaPipe requires Python ≤ 3.12.** This is the binding constraint.

On this dev machine, Python 3.12 and 3.14 are both installed:
```
py -0p
 -V:3.14[-64] *   C:\Users\DELL\AppData\Local\Python\pythoncore-3.14-64\python.exe
 -V:3.12[-64]     C:\Users\DELL\AppData\Local\Python\pythoncore-3.12-64\python.exe
```

We use `py -3.12` to create the project virtual environment.

---

## Development Environment (this machine — Intel x64)

### 1. Create virtual environment

```powershell
cd G:\snap\setusign
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 2. Install core dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run environment check

```powershell
python env/check_env.py
```

Expected: all packages installed, CPU execution provider only, no QNN EP (this is Intel hardware).

### 4. Set AI Hub token (optional on dev machine)

```powershell
$env:QAI_HUB_API_TOKEN = "your-token-here"
```

Or add to a `.env` file (gitignored) and load with `python-dotenv`.

---

## Target Environment (Snapdragon HP PC — ARM64)

### Architecture Notes

- Snapdragon X Elite/Plus runs Windows on ARM64.
- Python 3.12 ARM64 build is available from python.org.
- `onnxruntime-qnn` provides the QNN Execution Provider for Hexagon NPU access.
- MediaPipe provides ARM64 Windows wheels for 3.12.

### Installation

```powershell
# Use ARM64-native Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install ONNX Runtime with QNN EP
pip install onnxruntime-qnn

# Install remaining deps
pip install -r requirements.txt

# Verify QNN EP
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected output includes: QNNExecutionProvider
```

### Qualcomm AI Hub (model compilation)

AI Hub is used to compile/quantize models for the Hexagon NPU. The compilation
happens in the cloud via the `qai-hub` Python package:

```powershell
pip install qai-hub
qai-hub configure --api-token $env:QAI_HUB_API_TOKEN
```

**Note**: `qai-hub` may require x64 Python for some operations. If so, use a
separate x64 Python environment for export/compile only, and use the compiled
artifacts in the ARM64 runtime environment. This separation is documented per phase.

---

## Verified Package Compatibility (as of 2026-09-30)

| Package | Python 3.12 | Python 3.14 | ARM64 Windows | Source |
|---------|-------------|-------------|---------------|--------|
| onnxruntime | ✅ 1.30+ | ✅ 1.30+ | ✅ | [PyPI](https://pypi.org/project/onnxruntime/) |
| onnxruntime-qnn | ✅ | ❓ | ✅ (required) | [PyPI](https://pypi.org/project/onnxruntime-qnn/) |
| mediapipe | ✅ ≤0.10.x | ❌ | ✅ | [PyPI](https://pypi.org/project/mediapipe/), [GitHub](https://github.com/google-ai-edge/mediapipe) |
| torch | ✅ | ❓ | ❌ (use x64 emu) | [pytorch.org](https://pytorch.org/) |
| opencv-python | ✅ | ✅ | ✅ | [PyPI](https://pypi.org/project/opencv-python/) |
| numpy | ✅ | ✅ | ✅ | [PyPI](https://pypi.org/project/numpy/) |

---

## Two-Environment Strategy

For the target Snapdragon device, we may need two Python environments:

1. **Runtime env (ARM64 Python 3.12)**: runs the app, uses `onnxruntime-qnn` for NPU inference.
2. **Export/compile env (x64 Python 3.12, if needed)**: runs `qai-hub` CLI and PyTorch for model export.
   This runs under x64 emulation on ARM64 Windows (slower but functional).

On the Intel dev machine, a single environment suffices since everything runs on CPU.
