"""
SetuSign Environment Check
==========================
Reports system information, Python environment, installed ML packages,
execution providers, and AI Hub token status.

Hard rule #2: Always log the execution provider actually used.
Hard rule #4: Never print the API token value.
"""

import os
import sys
import platform
import json
import datetime
import subprocess


def get_windows_version():
    """Get detailed Windows version info."""
    ver = platform.version()
    release = platform.release()
    edition = platform.win32_edition() if hasattr(platform, 'win32_edition') else "unknown"
    return {
        "release": release,
        "version": ver,
        "edition": edition,
    }


def get_cpu_info():
    """Get CPU architecture and identity."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "is_arm64": platform.machine().lower() in ("aarch64", "arm64"),
    }


def get_python_info():
    """Get Python version and architecture details."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "architecture": platform.architecture()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
        "is_venv": sys.prefix != sys.base_prefix,
    }


def check_package(name):
    """Check if a package is installed and return its version."""
    try:
        import importlib.metadata
        version = importlib.metadata.version(name)
        return {"installed": True, "version": version}
    except Exception:
        return {"installed": False, "version": None}


def check_onnxruntime():
    """Check ONNX Runtime installation and available execution providers."""
    info = check_package("onnxruntime")
    if not info["installed"]:
        # Also check onnxruntime-qnn variant
        info = check_package("onnxruntime-qnn")
        if not info["installed"]:
            info["providers"] = []
            info["has_qnn"] = False
            info["note"] = "Neither onnxruntime nor onnxruntime-qnn is installed."
            return info

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        info["providers"] = providers
        info["has_qnn"] = "QNNExecutionProvider" in providers
        if info["has_qnn"]:
            info["note"] = "QNNExecutionProvider is available — Hexagon NPU inference supported."
        else:
            info["note"] = (
                "QNNExecutionProvider NOT available. "
                "Running on CPU only. "
                "Install onnxruntime-qnn on a Snapdragon device for NPU support."
            )
    except Exception as e:
        info["providers"] = []
        info["has_qnn"] = False
        info["note"] = f"Failed to query providers: {e}"

    return info


def check_mediapipe():
    """Check MediaPipe installation."""
    info = check_package("mediapipe")
    if info["installed"]:
        try:
            import mediapipe as mp
            info["note"] = "MediaPipe is installed and importable."
        except Exception as e:
            info["note"] = f"Installed but import failed: {e}"
    else:
        info["note"] = "MediaPipe not installed. Required for hand/pose keypoint extraction."
    return info


def check_torch():
    """Check PyTorch installation."""
    info = check_package("torch")
    if info["installed"]:
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
            info["note"] = f"PyTorch installed. CUDA: {'yes' if torch.cuda.is_available() else 'no'}."
        except Exception as e:
            info["note"] = f"Installed but import failed: {e}"
    else:
        info["note"] = "PyTorch not installed. Required for model training."
    return info


def check_ai_hub_token():
    """Check if the AI Hub API token environment variable is set (never print its value)."""
    token_var = "QAI_HUB_API_TOKEN"
    is_set = token_var in os.environ and len(os.environ[token_var].strip()) > 0
    return {
        "env_var": token_var,
        "is_set": is_set,
        "note": (
            f"{token_var} is set (value hidden)."
            if is_set
            else f"{token_var} is NOT set. Required for Qualcomm AI Hub model compilation."
        ),
    }


def check_opencv():
    """Check OpenCV installation."""
    info = check_package("opencv-python")
    if not info["installed"]:
        info = check_package("opencv-python-headless")
    if info["installed"]:
        try:
            import cv2
            info["version"] = cv2.__version__
            info["note"] = "OpenCV is installed and importable."
        except Exception as e:
            info["note"] = f"Installed but import failed: {e}"
    else:
        info["note"] = "OpenCV not installed. Required for video capture."
    return info


def main():
    """Run all environment checks and output results."""
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system": {
            "os": get_windows_version(),
            "cpu": get_cpu_info(),
        },
        "python": get_python_info(),
        "packages": {
            "onnxruntime": check_onnxruntime(),
            "mediapipe": check_mediapipe(),
            "torch": check_torch(),
            "opencv": check_opencv(),
        },
        "ai_hub_token": check_ai_hub_token(),
    }

    # Force UTF-8 output to avoid cp1252 encoding errors on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Pretty-print the report
    print("=" * 70)
    print("SetuSign Environment Check")
    print("=" * 70)

    print(f"\nTimestamp: {report['timestamp']}")

    # System
    sys_info = report["system"]
    print(f"\n--- System ---")
    print(f"  Windows {sys_info['os']['release']} (build {sys_info['os']['version']})")
    print(f"  Edition: {sys_info['os']['edition']}")
    print(f"  CPU arch: {sys_info['cpu']['machine']}")
    print(f"  Processor: {sys_info['cpu']['processor']}")
    print(f"  ARM64: {'YES' if sys_info['cpu']['is_arm64'] else 'NO'}")

    # Python
    py_info = report["python"]
    print(f"\n--- Python ---")
    print(f"  Version: {py_info['version']} ({py_info['architecture']})")
    print(f"  Implementation: {py_info['implementation']}")
    print(f"  Executable: {py_info['executable']}")
    print(f"  Virtual env: {'YES' if py_info['is_venv'] else 'NO'}")

    # Packages
    print(f"\n--- Packages ---")
    for name, info in report["packages"].items():
        status = f"v{info['version']}" if info["installed"] else "NOT INSTALLED"
        print(f"  {name}: {status}")
        if "note" in info:
            print(f"    -> {info['note']}")
        if name == "onnxruntime" and info.get("providers"):
            print(f"    -> Execution providers: {', '.join(info['providers'])}")

    # AI Hub Token
    token_info = report["ai_hub_token"]
    print(f"\n--- AI Hub ---")
    print(f"  {token_info['note']}")

    # Warnings
    print(f"\n--- Warnings ---")
    warnings = []
    if not sys_info["cpu"]["is_arm64"]:
        warnings.append(
            "This is NOT an ARM64 machine. QNN Execution Provider will not be available. "
            "All models will run on CPU. NPU testing requires a Snapdragon-powered device."
        )
    if py_info["version"].startswith("3.14") or py_info["version"].startswith("3.13"):
        warnings.append(
            f"Python {py_info['version']} detected. MediaPipe only supports <=3.12. "
            f"Use Python 3.12 in a virtual environment."
        )
    if not report["packages"]["onnxruntime"]["installed"]:
        warnings.append("ONNX Runtime is not installed. Run: pip install onnxruntime")
    if not report["packages"]["mediapipe"]["installed"]:
        warnings.append("MediaPipe is not installed. Run: pip install mediapipe")
    if not token_info["is_set"]:
        warnings.append(f"Set {token_info['env_var']} for Qualcomm AI Hub access.")

    if warnings:
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("  [ok] No warnings.")

    print(f"\n{'=' * 70}")

    # Save JSON report
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "check_env_output.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to: {json_path}")

    return report


if __name__ == "__main__":
    main()
