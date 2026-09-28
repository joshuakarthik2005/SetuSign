# SetuSign

> **Offline, two-way Indian Sign Language communicator for service counters,
> optimized for Snapdragon-powered HP PCs (Hexagon NPU).**

**Qualcomm Snapdragon® AI Lab Build & Present Challenge 2026** — Individual submission

---

## The Problem

Over **5 million deaf Indians** face daily communication barriers at service counters —
clinics, banks, government offices. Current solutions require internet connectivity,
violate privacy by streaming video to cloud servers, and don't support Indian Sign Language (ISL).

## The Solution

SetuSign is a **fully offline, privacy-first** two-way communicator:

| Direction | Pipeline |
|-----------|----------|
| **Sign → Text/Speech** | Camera → MediaPipe keypoints → Transformer recognizer → Text + TTS |
| **Speech → Captions** | Microphone → Whisper ASR → Large on-screen captions for deaf user |

### Key Innovations
- 🔒 **Privacy-first**: Processes skeleton keypoints only — raw video never stored or transmitted
- ⚡ **Real-time on Snapdragon NPU**: ONNX model via QNN Execution Provider, sub-50ms inference
- 🎯 **Confidence gate**: Below threshold shows top-3 candidates for one-tap confirmation
- 👤 **On-device personalization**: Frozen encoder + few-shot classifier = 3-5 samples per new sign
- 🌐 **Zero internet**: Works in rural areas with no connectivity

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SetuSign Pipeline                         │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────┐│
│  │  Webcam   │───>│  MediaPipe   │───>│  Transformer │───>│ TTS ││
│  │          │    │  Hands+Pose  │    │  Recognizer  │    │     ││
│  └──────────┘    │  (150 kpts)  │    │  (ONNX/QNN)  │    └─────┘│
│                  └──────────────┘    └──────────────┘           │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Mic    │───>│   Whisper    │───>│ Large Caption │           │
│  │          │    │   (ONNX)     │    │   Display     │           │
│  └──────────┘    └──────────────┘    └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
         All processing on-device via Snapdragon NPU
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd setusign

# 2. Create Python 3.12 venv
py -3.12 -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify environment
python env/check_env.py

# 5. Run the demo
python demo.py

# 6. View the presentation
start docs/index.html
```

---

## Technical Stack

| Component | Technology | Performance |
|-----------|-----------|-------------------|
| Keypoint Extraction | MediaPipe Hands + Pose | 30.6ms/frame, 33 FPS |
| Sign Recognition | 4-layer Transformer (2.2M params) | 6.0ms inference (ONNX, CPU) |
| Speech Recognition | Whisper-tiny | ONNX Runtime |
| Text-to-Speech | Windows SAPI (pyttsx3) | Native |
| Model Format | ONNX (2.6 MB FP32) | ~1.3 MB FP16 quantized |
| Runtime | ONNX Runtime 1.30 | QNNExecutionProvider |

---

## Dataset

**INCLUDE-50** (AI4Bharat, CC BY 4.0) — Indian Sign Language dataset

| Metric | Value |
|--------|-------|
| Videos | 958 (train: 689, val: 77, test: 192) |
| Classes | 50 ISL signs |
| Categories | 15 (Greetings, People, Places, etc.) |
| Published accuracy | 94.5% top-1 (Transformer baseline) |

---

## Repository Structure

```
setusign/
├── demo.py                    # Live webcam demo application
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── env/                       # Environment setup + check scripts
├── data/                      # Dataset splits, label maps, download scripts
│   ├── splits/                # Train/val/test split files
│   ├── label_maps/            # Class label mappings
│   └── download_include50.py  # Smart HTTP range-request downloader
├── src/setusign/              # Core application code
│   ├── keypoints/extract.py   # MediaPipe keypoint extraction pipeline
│   ├── recognizer/
│   │   ├── models.py          # Transformer + LSTM architectures
│   │   ├── dataset.py         # DataLoader with augmentations
│   │   └── train.py           # Training pipeline with ONNX export
│   ├── asr/                   # Whisper speech-to-text
│   ├── tts/                   # Text-to-speech
│   └── ui/                    # Desktop UI
├── models/                    # ONNX models (gitignored)
├── bench/                     # Benchmark scripts + results
│   └── results/               # Real benchmark measurements
├── docs/
│   └── index.html             # Interactive presentation page
└── tests/                     # Unit tests
```

---

## Hardware Requirements

| | Development | Target Deployment |
|--|-------------|-------------------|
| **CPU** | Any x64 (tested: Intel i5-1335U) | Snapdragon X Elite/Plus |
| **NPU** | Not required (CPU fallback) | Hexagon NPU via QNN EP |
| **OS** | Windows 11 | Windows on ARM |
| **Camera** | USB or built-in webcam | Same |
| **Python** | 3.12 (MediaPipe requirement) | 3.12 |

---

## Snapdragon NPU Deployment Path

1. **Model trained** in PyTorch on any machine
2. **Exported to ONNX** with opset 18
3. **Compiled for QNN** via Qualcomm AI Hub or AI Engine Direct SDK
4. **Runs on Hexagon NPU** via `QNNExecutionProvider` in ONNX Runtime
5. **Fallback to CPU** when NPU not available (development machines)

```python
# NPU-aware inference (automatic fallback)
providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("model.onnx", providers=providers)
```

---

## Third-Party Attribution

See [THIRD_PARTY.md](THIRD_PARTY.md) for full attribution.

Key dependencies:
- **INCLUDE dataset** — AI4Bharat (IIT Madras), CC BY 4.0
- **MediaPipe** — Google, Apache 2.0
- **ONNX Runtime** — Microsoft, MIT License
- **PyTorch** — Meta, BSD-3-Clause

---

## License

MIT License

Copyright (c) 2026 SetuSign Contributors
