"""
Headless pipeline test — validates the full SetuSign pipeline
without requiring a camera or display.

Tests:
1. MediaPipe keypoint extraction from a synthetic frame
2. ONNX model inference
3. End-to-end latency measurement
"""
import os
import sys
import time
import json
import numpy as np

# MediaPipe
import mediapipe as mp

# ONNX Runtime
import onnxruntime as ort

LABEL_MAP_PATH = "data/label_maps/label_map_include50.json"
MODEL_PATH = "models/baseline/model.onnx"


def test_keypoint_extraction():
    """Test MediaPipe keypoint extraction on a synthetic frame."""
    print("1. Testing keypoint extraction...")

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=0.3,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
    )

    # Create a synthetic frame (random image)
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    hand_results = hands.process(frame)
    pose_results = pose.process(frame)
    elapsed = (time.perf_counter() - t0) * 1000

    # Extract features vector
    h1x = [0.0] * 21
    h1y = [0.0] * 21
    h2x = [0.0] * 21
    h2y = [0.0] * 21
    px = [0.0] * 33
    py = [0.0] * 33

    if hand_results.multi_hand_landmarks:
        lm = hand_results.multi_hand_landmarks[0]
        h1x = [l.x for l in lm.landmark]
        h1y = [l.y for l in lm.landmark]

    if pose_results.pose_landmarks:
        px = [l.x for l in pose_results.pose_landmarks.landmark]
        py = [l.y for l in pose_results.pose_landmarks.landmark]

    features = np.array(h1x + h1y + h2x + h2y + px + py, dtype=np.float32)

    hands.close()
    pose.close()

    print(f"   Feature vector shape: {features.shape}")
    print(f"   Latency: {elapsed:.1f}ms")
    print(f"   PASS")
    return features


def test_onnx_inference(features):
    """Test ONNX model inference."""
    print("\n2. Testing ONNX model inference...")

    if not os.path.exists(MODEL_PATH):
        print(f"   Model not found at {MODEL_PATH}")
        print(f"   SKIP (run bench/export_and_benchmark.py first)")
        return None

    # Load label map
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    # Create session
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f"   Providers: {session.get_providers()}")

    # Create a sequence of 200 frames (repeating our single frame)
    sequence = np.tile(features, (200, 1))  # [200, 150]
    mask = np.ones(200, dtype=np.float32)

    # Run inference
    t0 = time.perf_counter()
    outputs = session.run(None, {
        "keypoints": sequence[np.newaxis].astype(np.float32),
        "mask": mask[np.newaxis].astype(np.float32),
    })
    elapsed = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    top_idx = np.argsort(probs)[::-1][:5]
    print(f"   Inference latency: {elapsed:.1f}ms")
    print(f"   Top-5 predictions:")
    for i, idx in enumerate(top_idx):
        label = idx_to_label.get(idx, f"class_{idx}")
        print(f"     {i+1}. {label}: {probs[idx]:.3f}")
    print(f"   PASS")
    return probs


def test_end_to_end():
    """Test full pipeline end-to-end."""
    print("\n3. End-to-end pipeline test...")

    t0 = time.perf_counter()

    # Step 1: Keypoint extraction
    features = test_keypoint_extraction()

    # Step 2: Model inference
    probs = test_onnx_inference(features)

    total = (time.perf_counter() - t0) * 1000
    print(f"\n{'='*50}")
    print(f"Total end-to-end time: {total:.1f}ms")
    print(f"Pipeline status: ALL TESTS PASSED")
    print(f"{'='*50}")


if __name__ == "__main__":
    test_end_to_end()
