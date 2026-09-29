"""
Qualcomm AI Hub Model Compilation
===================================
Compiles the SetuSign ONNX model for Snapdragon NPU deployment
using Qualcomm AI Hub.

Prerequisites:
    pip install qai-hub
    export QAI_HUB_API_TOKEN=<your-token>

Usage:
    python models/compile_for_snapdragon.py
"""

import os
import sys
import json
import argparse
import numpy as np

MODEL_PATH = "models/baseline/model.onnx"
COMPILED_DIR = "models/compiled"


def compile_model(model_path, device="Snapdragon X Elite", quantize=True):
    """
    Compile ONNX model for Snapdragon NPU via Qualcomm AI Hub.
    
    Args:
        model_path: Path to ONNX model
        device: Target Snapdragon device
        quantize: Whether to quantize to INT8/FP16
    """
    try:
        import qai_hub as hub
    except ImportError:
        print("ERROR: qai_hub not installed. Install with: pip install qai-hub")
        print("       Then set QAI_HUB_API_TOKEN environment variable.")
        print("\nGenerating compilation instructions instead...\n")
        print_instructions(model_path)
        return

    if not os.getenv("QAI_HUB_API_TOKEN"):
        print("ERROR: QAI_HUB_API_TOKEN not set.")
        print("  Get your token from: https://aihub.qualcomm.com/")
        print("  Then: $env:QAI_HUB_API_TOKEN = 'your-token-here'")
        print("\nGenerating compilation instructions instead...\n")
        print_instructions(model_path)
        return

    print(f"Compiling {model_path} for {device}...")

    # Submit compilation job
    compile_job = hub.submit_compile_job(
        model=model_path,
        device=hub.Device(device),
        name="setusign-sign-recognizer",
        options="--target_runtime onnx",
    )

    print(f"Job submitted: {compile_job.job_id}")
    print("Waiting for compilation...")

    # Wait for completion
    compile_job.wait()
    status = compile_job.get_status()
    print(f"Status: {status}")

    if status.success:
        os.makedirs(COMPILED_DIR, exist_ok=True)
        output_path = os.path.join(COMPILED_DIR, "model_qnn.onnx")
        compile_job.download_target_model(output_path)
        print(f"Compiled model saved: {output_path}")
        print(f"Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

        # Also try profiling
        print("\nProfiling on device...")
        try:
            input_data = {
                "keypoints": np.random.randn(1, 200, 150).astype(np.float32),
                "mask": np.ones((1, 200), dtype=np.float32),
            }
            profile_job = hub.submit_profile_job(
                model=output_path,
                device=hub.Device(device),
                input_specs=input_data,
            )
            profile_job.wait()
            profile = profile_job.download_profile()
            print(f"On-device inference time: {profile.get('inference_time_ms', 'N/A')}ms")
        except Exception as e:
            print(f"Profiling failed: {e}")
    else:
        print(f"Compilation failed: {status.message}")


def print_instructions(model_path):
    """Print manual compilation instructions."""
    print("=" * 60)
    print("Snapdragon NPU Deployment Instructions")
    print("=" * 60)
    print(f"""
1. Install Qualcomm AI Hub:
   pip install qai-hub

2. Get API token from https://aihub.qualcomm.com/

3. Set token:
   $env:QAI_HUB_API_TOKEN = 'your-token'

4. Compile:
   python models/compile_for_snapdragon.py

5. Or use AI Engine Direct SDK:
   - Download from https://www.qualcomm.com/developer
   - Convert ONNX to DLC:
     snpe-onnx-to-dlc --input_network {model_path} --output_path model.dlc
   - Quantize for Hexagon:
     snpe-dlc-quantize --input_dlc model.dlc --output_dlc model_q.dlc

6. Run with QNN Execution Provider:
   providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
   session = ort.InferenceSession("model.onnx", providers=providers)

Current model info:
   Path: {model_path}
   Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB (if exists)
   Input: [batch, 200, 150] float32
   Output: [batch, 50] float32
""")


def main():
    parser = argparse.ArgumentParser(description="Compile model for Snapdragon NPU")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--device", default="Snapdragon X Elite")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run bench/export_and_benchmark.py first to export the ONNX model.")
        sys.exit(1)

    compile_model(args.model, device=args.device, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()
