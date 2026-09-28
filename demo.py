"""
SetuSign Demo Application
==========================
Two-way ISL communicator for service counters.
Runs entirely on-device with ONNX Runtime (Snapdragon NPU-ready).

Usage:
    python demo.py
"""

import os
import sys
import time
import json
import threading
import queue
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import onnxruntime as ort

# ── Configuration ──────────────────────────────────────────────────────────
LABEL_MAP_PATH = "data/label_maps/label_map_include50.json"
MODEL_PATH = "models/baseline/model.onnx"
MAX_SEQ_LEN = 60  # frames to accumulate before classification
CONFIDENCE_THRESHOLD = 0.6
TOP_K = 3
WINDOW_NAME = "SetuSign - ISL Communicator"


class KeypointExtractor:
    """Real-time MediaPipe keypoint extraction."""

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

    def extract(self, frame_rgb):
        """Extract 150-dim keypoint vector from a single frame."""
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        # Hands: 2 x 21 x 2 = 84
        h1x, h1y = [0.0] * 21, [0.0] * 21
        h2x, h2y = [0.0] * 21, [0.0] * 21

        if hand_results.multi_hand_landmarks:
            for idx, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                x = [l.x for l in hand_lm.landmark]
                y = [l.y for l in hand_lm.landmark]
                # Use handedness to assign consistently
                if hand_results.multi_handedness:
                    label = hand_results.multi_handedness[idx].classification[0].label
                    if label == "Right":
                        h1x, h1y = x, y
                    else:
                        h2x, h2y = x, y
                elif idx == 0:
                    h1x, h1y = x, y
                else:
                    h2x, h2y = x, y

        # Pose: 33 x 2 = 66
        px, py = [0.0] * 33, [0.0] * 33
        if pose_results.pose_landmarks:
            px = [l.x for l in pose_results.pose_landmarks.landmark]
            py = [l.y for l in pose_results.pose_landmarks.landmark]

        # Concatenate: [h1x(21), h1y(21), h2x(21), h2y(21), px(33), py(33)] = 150
        features = h1x + h1y + h2x + h2y + px + py
        return np.array(features, dtype=np.float32)

    def draw_landmarks(self, frame, frame_rgb):
        """Draw hand and pose landmarks on frame for visualization."""
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1),
                )

        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 128, 0), thickness=2),
                self.mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=1),
            )

        hands_detected = hand_results.multi_hand_landmarks is not None
        return frame, hands_detected


class SignRecognizer:
    """ONNX-based sign language recognizer (NPU-ready)."""

    def __init__(self, model_path, label_map, providers=None):
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.num_classes = len(label_map)

        if providers is None:
            # Try QNN first (Snapdragon NPU), fall back to CPU
            available = ort.get_available_providers()
            if "QNNExecutionProvider" in available:
                providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        if os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.has_model = True
            print(f"Model loaded: {model_path}")
            print(f"  Providers: {self.session.get_providers()}")
        else:
            self.session = None
            self.has_model = False
            print(f"No model at {model_path} — running in keypoint-only mode")

    def predict(self, keypoint_sequence):
        """
        Classify a keypoint sequence.
        
        Args:
            keypoint_sequence: [T, 150] numpy array
        
        Returns:
            list of (label, confidence) tuples, sorted by confidence
        """
        if not self.has_model:
            return self._demo_predict(keypoint_sequence)

        # Pad/truncate to model's expected length (fixed at 200)
        T, F = keypoint_sequence.shape
        max_len = 200
        if T > max_len:
            indices = np.linspace(0, T - 1, max_len, dtype=int)
            keypoint_sequence = keypoint_sequence[indices]
            T = max_len
        elif T < max_len:
            pad = np.zeros((max_len - T, F), dtype=np.float32)
            keypoint_sequence = np.concatenate([keypoint_sequence, pad], axis=0)

        # Create mask
        mask = np.zeros(max_len, dtype=np.float32)
        mask[:min(T, max_len)] = 1.0

        # Run inference
        inputs = {
            "keypoints": keypoint_sequence[np.newaxis],  # [1, T, 150]
            "mask": mask[np.newaxis],  # [1, T]
        }
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]  # [num_classes]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Sort by confidence
        results = []
        for idx in np.argsort(probs)[::-1]:
            label = self.idx_to_label.get(idx, f"class_{idx}")
            results.append((label, float(probs[idx])))

        return results

    def _demo_predict(self, keypoint_sequence):
        """Demo prediction when no model is loaded — uses hand activity heuristic."""
        # Compute hand activity (non-zero keypoints)
        T = keypoint_sequence.shape[0]
        h1_activity = np.mean(np.abs(keypoint_sequence[:, :42]) > 0.01)
        h2_activity = np.mean(np.abs(keypoint_sequence[:, 42:84]) > 0.01)

        # Simple heuristic for demo
        if h1_activity > 0.5 and h2_activity > 0.5:
            return [("hello", 0.7), ("thankyou", 0.15), ("goodmorning", 0.1)]
        elif h1_activity > 0.3:
            return [("hello", 0.6), ("good", 0.2), ("yes", 0.1)]
        else:
            return [("waiting...", 0.0)]


class TextToSpeech:
    """Threaded TTS to avoid blocking the main loop."""

    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.last_spoken = ""
        self.last_spoken_time = 0

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        while True:
            text = self.queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

    def speak(self, text, cooldown=3.0):
        """Speak text if not recently spoken."""
        now = time.time()
        if text != self.last_spoken or (now - self.last_spoken_time) > cooldown:
            self.last_spoken = text
            self.last_spoken_time = now
            self.queue.put(text)


def draw_ui(frame, recognized_text, confidence, top_k_results,
            hands_detected, recording, frame_count, caption_text=""):
    """Draw the service counter UI overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 200), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Status bar at top
    status_color = (0, 255, 128) if hands_detected else (100, 100, 100)
    status_text = "HANDS DETECTED" if hands_detected else "Show your hands to start signing"
    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.putText(frame, f"SetuSign | {status_text}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 30, 20), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"REC {frame_count}", (w - 120, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Recognition result panel (bottom)
    y_base = h - 180

    if recognized_text and confidence > 0:
        # Main result
        conf_color = (0, 255, 128) if confidence >= CONFIDENCE_THRESHOLD else (0, 200, 255)
        cv2.putText(frame, f"Recognized: {recognized_text.upper()}",
                    (20, y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.1%}",
                    (20, y_base + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

        # Top-K candidates
        if confidence < CONFIDENCE_THRESHOLD and top_k_results:
            cv2.putText(frame, "Top candidates (tap to confirm):",
                        (20, y_base + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            for i, (label, conf) in enumerate(top_k_results[:TOP_K]):
                bar_w = int(conf * 300)
                y = y_base + 125 + i * 25
                cv2.rectangle(frame, (20, y), (20 + bar_w, y + 18), conf_color, -1)
                cv2.putText(frame, f"{i+1}. {label} ({conf:.1%})",
                            (25, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Sign a word to begin...",
                    (20, y_base + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

    # Caption area for speech-to-text (right side)
    if caption_text:
        # Large caption box for deaf user
        cv2.rectangle(frame, (w // 2 + 20, h - 200), (w - 10, h - 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (w // 2 + 20, h - 200), (w - 10, h - 10), (0, 200, 255), 2)
        cv2.putText(frame, "CAPTIONS:", (w // 2 + 30, h - 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        # Word wrap captions
        words = caption_text.split()
        line = ""
        y_cap = h - 150
        for word in words:
            test = line + " " + word if line else word
            if len(test) > 30:
                cv2.putText(frame, line, (w // 2 + 30, y_cap),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_cap += 30
                line = word
            else:
                line = test
        if line:
            cv2.putText(frame, line, (w // 2 + 30, y_cap),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "Q: Quit | R: Reset | S: Speak result",
                (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="SetuSign Demo")
    parser.add_argument("--model", default=MODEL_PATH, help="ONNX model path")
    parser.add_argument("--label-map", default=LABEL_MAP_PATH)
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    # Load label map
    with open(args.label_map) as f:
        label_map = json.load(f)
    print(f"Loaded {len(label_map)} sign classes")

    # Initialize components
    extractor = KeypointExtractor()
    recognizer = SignRecognizer(args.model, label_map)
    tts = TextToSpeech()

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    print(f"\nCamera opened ({args.width}x{args.height})")
    print("Press Q to quit, R to reset, S to speak result")
    print("=" * 50)

    # State
    keypoint_buffer = []
    recognized_text = ""
    confidence = 0.0
    top_k_results = []
    recording = False
    fps_counter = []

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract keypoints and draw landmarks
        kp_vector = extractor.extract(frame_rgb)
        frame, hands_detected = extractor.draw_landmarks(frame, frame_rgb)

        # Buffer keypoints when hands are detected
        if hands_detected:
            keypoint_buffer.append(kp_vector)
            recording = True

            # Classify when we have enough frames
            if len(keypoint_buffer) >= MAX_SEQ_LEN:
                sequence = np.array(keypoint_buffer, dtype=np.float32)
                results = recognizer.predict(sequence)

                if results and results[0][1] > 0:
                    recognized_text = results[0][0]
                    confidence = results[0][1]
                    top_k_results = results[:TOP_K]

                    # Auto-speak if confident
                    if confidence >= CONFIDENCE_THRESHOLD:
                        tts.speak(recognized_text)

                # Keep last half of buffer for sliding window
                keypoint_buffer = keypoint_buffer[MAX_SEQ_LEN // 2:]

        elif recording and len(keypoint_buffer) > 10:
            # Hands disappeared — classify what we have
            sequence = np.array(keypoint_buffer, dtype=np.float32)
            results = recognizer.predict(sequence)

            if results and results[0][1] > 0:
                recognized_text = results[0][0]
                confidence = results[0][1]
                top_k_results = results[:TOP_K]

                if confidence >= CONFIDENCE_THRESHOLD:
                    tts.speak(recognized_text)

            keypoint_buffer = []
            recording = False
        elif not hands_detected:
            recording = False

        # Calculate FPS
        elapsed = time.time() - t0
        fps_counter.append(elapsed)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 120, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

        # Draw UI
        frame = draw_ui(frame, recognized_text, confidence, top_k_results,
                        hands_detected, recording, len(keypoint_buffer))

        cv2.imshow(WINDOW_NAME, frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            keypoint_buffer = []
            recognized_text = ""
            confidence = 0.0
            top_k_results = []
            recording = False
        elif key == ord("s") and recognized_text:
            tts.speak(recognized_text, cooldown=0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
