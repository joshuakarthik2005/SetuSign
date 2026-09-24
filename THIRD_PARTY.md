# Third-Party Attribution

This project uses the following third-party datasets, models, and libraries.

---

## Datasets

### INCLUDE-50 (AI4Bharat)
- **Description**: Subset of the INCLUDE dataset containing 50 Indian Sign Language (ISL) signs,
  with video recordings from multiple signers.
- **Source**: https://ai4bharat.iitm.ac.in/include
- **Paper**: Sridhar, A., et al. "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition."
  ACM Multimedia 2020.
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Usage**: Training and evaluation of sign recognition models.

---

## Models

### MediaPipe Hands & Pose
- **Description**: Google's hand and body pose landmark detection models.
- **Source**: https://github.com/google-ai-edge/mediapipe
- **License**: Apache License 2.0
- **Usage**: Extracting hand and pose keypoints from video frames.

### ONNX Runtime
- **Description**: Cross-platform inference engine with QNN Execution Provider for Snapdragon NPUs.
- **Source**: https://github.com/microsoft/onnxruntime
- **License**: MIT License
- **Usage**: On-device model inference.

---

## Libraries

| Library | License | Usage |
|---------|---------|-------|
| MediaPipe | Apache 2.0 | Hand/pose landmark extraction |
| ONNX Runtime | MIT | Model inference (CPU/QNN EP) |
| NumPy | BSD 3-Clause | Numerical computation |
| OpenCV | Apache 2.0 | Video capture and processing |
| PyTorch | BSD 3-Clause | Model training (development only) |
| Qualcomm AI Hub | Proprietary | Model compilation for Snapdragon |

---

## Fonts & Assets

_To be added as UI development progresses._
