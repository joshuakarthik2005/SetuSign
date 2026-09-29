"""
SetuSign Full Demo — Two-Way ISL Communicator
===============================================
Complete service counter demo with:
  - Sign -> Text/Speech (webcam + MediaPipe + ONNX + TTS)
  - Speech -> Captions (Whisper ASR + large display)
  - On-device personalization (few-shot enrollment)
  
Usage:
    python demo_full.py
    python demo_full.py --no-asr          # Skip Whisper (faster startup)
    python demo_full.py --enroll hello    # Start enrollment for "hello"
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

sys.path.insert(0, str(Path(__file__).parent))
from src.setusign.personalization.engine import PersonalizationManager

# Try to import ASR
try:
    from src.setusign.asr.recognizer import CaptionService
    HAS_ASR = True
except ImportError:
    HAS_ASR = False

LABEL_MAP_PATH = "data/label_maps/label_map_include50.json"
MODEL_PATH = "models/baseline/model.onnx"
CONFIDENCE_THRESHOLD = 0.6
MAX_SEQ_LEN = 60
TOP_K = 3

# ── Color Palette ─────────────────────────────────────────────
COLORS = {
    "bg_dark": (20, 20, 25),
    "bg_panel": (30, 30, 38),
    "accent": (241, 102, 99),      # Indigo-ish (BGR)
    "green": (128, 255, 0),
    "cyan": (212, 182, 6),
    "orange": (11, 158, 245),
    "white": (240, 240, 232),
    "dim": (160, 136, 136),
    "red": (60, 60, 255),
}


class KeypointExtractor:
    """Real-time MediaPipe keypoint extraction."""

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

    def extract(self, frame_rgb):
        """Extract 150-dim keypoint vector from a single frame."""
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        h1x, h1y = [0.0]*21, [0.0]*21
        h2x, h2y = [0.0]*21, [0.0]*21

        if hand_results.multi_hand_landmarks:
            for idx, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                x = [l.x for l in hand_lm.landmark]
                y = [l.y for l in hand_lm.landmark]
                if hand_results.multi_handedness:
                    label = hand_results.multi_handedness[idx].classification[0].label
                    if label == "Right": h1x, h1y = x, y
                    else: h2x, h2y = x, y
                elif idx == 0: h1x, h1y = x, y
                else: h2x, h2y = x, y

        px, py = [0.0]*33, [0.0]*33
        if pose_results.pose_landmarks:
            px = [l.x for l in pose_results.pose_landmarks.landmark]
            py = [l.y for l in pose_results.pose_landmarks.landmark]

        return np.array(h1x+h1y+h2x+h2y+px+py, dtype=np.float32)

    def draw_landmarks(self, frame, frame_rgb):
        """Draw landmarks and return (frame, hands_detected)."""
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,128), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(0,200,255), thickness=1),
                )
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255,128,0), thickness=2),
                self.mp_drawing.DrawingSpec(color=(255,200,0), thickness=1),
            )
        return frame, hand_results.multi_hand_landmarks is not None


class SignRecognizer:
    """ONNX-based sign recognizer."""

    def __init__(self, model_path, label_map):
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.session = None

        if os.path.exists(model_path):
            providers = ["CPUExecutionProvider"]
            available = ort.get_available_providers()
            if "QNNExecutionProvider" in available:
                providers = ["QNNExecutionProvider"] + providers
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"Model: {model_path} ({providers[0]})")

    def predict(self, keypoint_seq):
        """Classify keypoint sequence, returns [(label, conf), ...]."""
        if self.session is None:
            return [("no_model", 0.0)]

        T, F = keypoint_seq.shape
        max_len = 200
        if T > max_len:
            idx = np.linspace(0, T-1, max_len, dtype=int)
            keypoint_seq = keypoint_seq[idx]
            T = max_len
        elif T < max_len:
            keypoint_seq = np.concatenate([keypoint_seq, np.zeros((max_len-T, F), dtype=np.float32)])

        mask = np.zeros(max_len, dtype=np.float32)
        mask[:min(T, max_len)] = 1.0

        outputs = self.session.run(None, {
            "keypoints": keypoint_seq[np.newaxis],
            "mask": mask[np.newaxis],
        })
        logits = outputs[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / exp_l.sum()

        results = []
        for i in np.argsort(probs)[::-1]:
            results.append((self.idx_to_label.get(i, f"class_{i}"), float(probs[i])))
        return results


class TextToSpeech:
    """Threaded TTS."""
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.last_spoken = ""
        self.last_time = 0

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        while True:
            text = self.q.get()
            if text is None: break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception: pass

    def speak(self, text, cooldown=3.0):
        now = time.time()
        if text != self.last_spoken or (now - self.last_time) > cooldown:
            self.last_spoken = text
            self.last_time = now
            self.q.put(text)


def draw_premium_ui(frame, state):
    """Draw premium two-panel service counter UI."""
    h, w = frame.shape[:2]
    split_x = int(w * 0.6)  # 60/40 split

    # ── Left panel: Sign recognition ──
    # Top bar
    cv2.rectangle(frame, (0, 0), (split_x, 45), COLORS["bg_dark"], -1)
    status_color = COLORS["green"] if state["hands"] else COLORS["dim"]
    cv2.circle(frame, (20, 22), 6, status_color, -1)
    cv2.putText(frame, "SIGN RECOGNITION", (35, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["white"], 2)

    # FPS
    cv2.putText(frame, f"{state['fps']:.0f} FPS", (split_x - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["green"], 1)

    # Recording indicator
    if state["recording"]:
        cv2.circle(frame, (split_x - 120, 22), 6, COLORS["red"], -1)

    # Bottom recognition panel
    panel_h = 160
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-panel_h), (split_x, h), COLORS["bg_dark"], -1)
    frame[h-panel_h:h, 0:split_x] = cv2.addWeighted(
        overlay[h-panel_h:h, 0:split_x], 0.88,
        frame[h-panel_h:h, 0:split_x], 0.12, 0)

    y_base = h - panel_h + 15
    if state["text"] and state["confidence"] > 0:
        conf_color = COLORS["green"] if state["confidence"] >= CONFIDENCE_THRESHOLD else COLORS["orange"]

        # Main result with large text
        cv2.putText(frame, state["text"].upper(), (15, y_base + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, conf_color, 3)

        # Confidence bar
        bar_w = int(state["confidence"] * (split_x - 40))
        cv2.rectangle(frame, (15, y_base + 50), (15 + bar_w, y_base + 60), conf_color, -1)
        cv2.putText(frame, f"{state['confidence']:.0%}", (split_x - 60, y_base + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)

        # Top-K if low confidence
        if state["confidence"] < CONFIDENCE_THRESHOLD and state["top_k"]:
            cv2.putText(frame, "Select:", (15, y_base + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["dim"], 1)
            for i, (lbl, conf) in enumerate(state["top_k"][:3]):
                y = y_base + 105 + i * 22
                cv2.putText(frame, f"[{i+1}] {lbl} ({conf:.0%})", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)
    else:
        cv2.putText(frame, "Show your hands to start signing...",
                    (15, y_base + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["dim"], 1)

    # Enrollment indicator
    if state.get("enrolling"):
        cv2.rectangle(frame, (0, 46), (split_x, 80), (0, 0, 80), -1)
        cv2.putText(frame, f"ENROLLING: {state['enroll_label']} ({state['enroll_count']}/{state['enroll_min']})",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["orange"], 2)

    # ── Right panel: Captions ──
    cv2.rectangle(frame, (split_x, 0), (w, h), COLORS["bg_panel"], -1)

    # Caption header
    cv2.rectangle(frame, (split_x, 0), (w, 45), COLORS["bg_dark"], -1)
    cv2.putText(frame, "CAPTIONS FOR YOU", (split_x + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["cyan"], 2)

    # Caption text (large, readable)
    caption = state.get("caption", "Waiting for speech...")
    # Word wrap
    words = caption.split()
    lines = []
    line = ""
    max_chars = 22
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line = f"{line} {word}" if line else word
        else:
            if line: lines.append(line)
            line = word
    if line: lines.append(line)

    for i, ln in enumerate(lines[:8]):
        y = 90 + i * 50
        cv2.putText(frame, ln, (split_x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS["white"], 2)

    # Caption history
    history = state.get("caption_history", [])
    if history:
        y_hist = h - 150
        cv2.putText(frame, "Recent:", (split_x + 15, y_hist),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["dim"], 1)
        for i, hist_text in enumerate(history[-3:]):
            cv2.putText(frame, hist_text[:30], (split_x + 15, y_hist + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["dim"], 1)

    # Controls bar at bottom
    cv2.rectangle(frame, (0, h-20), (w, h), COLORS["bg_dark"], -1)
    controls = "Q:Quit  R:Reset  S:Speak  E:Enroll  P:Save Profile"
    cv2.putText(frame, controls, (10, h-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["dim"], 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="SetuSign Full Demo")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--label-map", default=LABEL_MAP_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-asr", action="store_true", help="Disable speech recognition")
    parser.add_argument("--enroll", type=str, default=None, help="Start enrollment for a sign")
    parser.add_argument("--profile", default="default", help="User profile name")
    args = parser.parse_args()

    # Load label map
    with open(args.label_map) as f:
        label_map = json.load(f)

    # Initialize components
    print("Initializing SetuSign...")
    extractor = KeypointExtractor()
    recognizer = SignRecognizer(args.model, label_map)
    tts = TextToSpeech()

    # ONNX session for personalization
    onnx_session = recognizer.session
    personalization = PersonalizationManager(
        onnx_session=onnx_session, embed_dim=50, profile_dir="data/profiles"
    )
    personalization.load_profile(args.profile)

    # ASR
    caption_service = None
    if HAS_ASR and not args.no_asr:
        print("Starting speech recognition...")
        caption_service = CaptionService(model_size="tiny", language="en")
        caption_service.start()

    # Camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    print(f"Camera: {args.width}x{args.height}")
    print("Controls: Q=Quit, R=Reset, S=Speak, E=Enroll, P=Save Profile")
    print("=" * 50)

    # Start enrollment if requested
    if args.enroll:
        personalization.start_enrollment(args.enroll)

    # State
    kp_buffer = []
    recognized_text = ""
    confidence = 0.0
    top_k_results = []
    recording = False
    fps_times = []

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Keypoints
        kp_vector = extractor.extract(frame_rgb)
        frame, hands_detected = extractor.draw_landmarks(frame, frame_rgb)

        # Buffer keypoints
        if hands_detected:
            kp_buffer.append(kp_vector)
            recording = True

            if len(kp_buffer) >= MAX_SEQ_LEN:
                seq = np.array(kp_buffer, dtype=np.float32)

                # Enrollment mode
                if personalization.enrolling:
                    personalization.add_enrollment_sample(seq)
                    kp_buffer = []
                    recording = False
                else:
                    # Recognition
                    results = recognizer.predict(seq)
                    if results and results[0][1] > 0:
                        recognized_text = results[0][0]
                        confidence = results[0][1]
                        top_k_results = results[:TOP_K]
                        if confidence >= CONFIDENCE_THRESHOLD:
                            tts.speak(recognized_text)
                    kp_buffer = kp_buffer[MAX_SEQ_LEN // 2:]

        elif recording and len(kp_buffer) > 10:
            seq = np.array(kp_buffer, dtype=np.float32)
            if personalization.enrolling:
                personalization.add_enrollment_sample(seq)
            else:
                results = recognizer.predict(seq)
                if results and results[0][1] > 0:
                    recognized_text = results[0][0]
                    confidence = results[0][1]
                    top_k_results = results[:TOP_K]
                    if confidence >= CONFIDENCE_THRESHOLD:
                        tts.speak(recognized_text)
            kp_buffer = []
            recording = False

        # FPS
        elapsed = time.time() - t0
        fps_times.append(elapsed)
        if len(fps_times) > 30: fps_times.pop(0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

        # Caption
        caption = "Waiting for speech..."
        caption_history = []
        if caption_service:
            caption = caption_service.get_caption() or "Waiting for speech..."
            caption_history = caption_service.get_history()

        # Draw UI
        ui_state = {
            "hands": hands_detected,
            "recording": recording,
            "text": recognized_text,
            "confidence": confidence,
            "top_k": top_k_results,
            "fps": fps,
            "caption": caption,
            "caption_history": caption_history,
            "enrolling": personalization.enrolling,
            "enroll_label": personalization.enroll_label or "",
            "enroll_count": len(personalization.enroll_samples),
            "enroll_min": personalization.min_samples,
        }
        frame = draw_premium_ui(frame, ui_state)

        cv2.imshow("SetuSign - ISL Communicator", frame)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            kp_buffer = []
            recognized_text = ""
            confidence = 0.0
            top_k_results = []
            recording = False
        elif key == ord("s") and recognized_text:
            tts.speak(recognized_text, cooldown=0)
        elif key == ord("e"):
            if personalization.enrolling:
                personalization.finish_enrollment()
            else:
                label = input("Enter sign name to enroll: ").strip()
                if label:
                    personalization.start_enrollment(label)
        elif key == ord("p"):
            personalization.save_profile(args.profile)
        elif key in [ord("1"), ord("2"), ord("3")]:
            idx = key - ord("1")
            if top_k_results and idx < len(top_k_results):
                recognized_text = top_k_results[idx][0]
                confidence = 1.0
                tts.speak(recognized_text, cooldown=0)

    cap.release()
    cv2.destroyAllWindows()
    if caption_service:
        caption_service.stop()


if __name__ == "__main__":
    main()
