"""
MediaPipe Keypoint Extraction for INCLUDE-50
=============================================
Extracts hand (2x21 = 42 points) and pose (33 points) keypoints from
sign language videos using MediaPipe. Saves as compact .npz files.

Output format per video:
  - hand1_x, hand1_y: [T, 21] arrays (right hand)
  - hand2_x, hand2_y: [T, 21] arrays (left hand)
  - pose_x, pose_y: [T, 33] arrays (body pose)
  - label: string label from path

Total keypoints per frame: 2*21 + 33 = 75 (x,y) = 150 values
This matches the AI4Bharat approach but uses modern MediaPipe Holistic.
"""

import os
import sys
import json
import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def extract_keypoints_from_video(video_path, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5):
    """
    Extract hand and pose keypoints from a video file.
    
    Returns dict with arrays of shape [T, N] for each keypoint group,
    where T = number of frames, N = number of landmarks.
    Returns None if video can't be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Storage
    hand1_x, hand1_y = [], []
    hand2_x, hand2_y = [], []
    pose_x, pose_y = [], []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        hand_results = hands.process(frame_rgb)
        h1x, h1y = [0.0] * 21, [0.0] * 21
        h2x, h2y = [0.0] * 21, [0.0] * 21

        if hand_results.multi_hand_landmarks:
            if len(hand_results.multi_hand_landmarks) > 0:
                lm = hand_results.multi_hand_landmarks[0]
                h1x = [l.x for l in lm.landmark]
                h1y = [l.y for l in lm.landmark]
            if len(hand_results.multi_hand_landmarks) > 1:
                lm = hand_results.multi_hand_landmarks[1]
                h2x = [l.x for l in lm.landmark]
                h2y = [l.y for l in lm.landmark]

            # Swap hands based on wrist proximity to pose wrists
            # (consistent left/right assignment)
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) >= 2:
                # Use handedness info if available
                if hand_results.multi_handedness:
                    for idx, handedness in enumerate(hand_results.multi_handedness):
                        label = handedness.classification[0].label
                        lm = hand_results.multi_hand_landmarks[idx]
                        if label == "Right":
                            h1x = [l.x for l in lm.landmark]
                            h1y = [l.y for l in lm.landmark]
                        else:
                            h2x = [l.x for l in lm.landmark]
                            h2y = [l.y for l in lm.landmark]

        hand1_x.append(h1x)
        hand1_y.append(h1y)
        hand2_x.append(h2x)
        hand2_y.append(h2y)

        # Process pose
        pose_results = pose.process(frame_rgb)
        px, py = [0.0] * 33, [0.0] * 33
        if pose_results.pose_landmarks:
            px = [l.x for l in pose_results.pose_landmarks.landmark]
            py = [l.y for l in pose_results.pose_landmarks.landmark]

        pose_x.append(px)
        pose_y.append(py)
        frame_count += 1

    cap.release()
    hands.close()
    pose.close()

    if frame_count == 0:
        return None

    return {
        "hand1_x": np.array(hand1_x, dtype=np.float32),  # [T, 21]
        "hand1_y": np.array(hand1_y, dtype=np.float32),
        "hand2_x": np.array(hand2_x, dtype=np.float32),
        "hand2_y": np.array(hand2_y, dtype=np.float32),
        "pose_x": np.array(pose_x, dtype=np.float32),    # [T, 33]
        "pose_y": np.array(pose_y, dtype=np.float32),
        "fps": fps,
        "num_frames": frame_count,
    }


def keypoints_to_feature_vector(kp_dict):
    """
    Convert keypoints dict to a single feature array of shape [T, 150].
    
    Layout: [hand1_x(21), hand1_y(21), hand2_x(21), hand2_y(21),
             pose_x(33), pose_y(33)] = 150 features per frame.
    
    This is compatible with the AI4Bharat INCLUDE baseline (134 features
    uses a subset of pose; we use all 33 pose landmarks = 150 total).
    """
    features = np.concatenate([
        kp_dict["hand1_x"],  # [T, 21]
        kp_dict["hand1_y"],
        kp_dict["hand2_x"],
        kp_dict["hand2_y"],
        kp_dict["pose_x"],   # [T, 33]
        kp_dict["pose_y"],
    ], axis=1)  # [T, 150]
    return features


def process_split(split_file, video_dir, output_dir, label_map=None):
    """
    Process all videos in a split file.
    
    Args:
        split_file: Path to split .txt file with relative video paths
        video_dir: Root directory containing the videos
        output_dir: Directory to save .npz keypoint files
        label_map: Optional dict mapping sign names to integer labels
    
    Returns:
        dict with stats
    """
    with open(split_file) as f:
        video_paths = [l.strip() for l in f if l.strip()]

    os.makedirs(output_dir, exist_ok=True)

    stats = {"total": len(video_paths), "success": 0, "failed": 0, "skipped": 0}
    failed_videos = []

    for vpath in tqdm(video_paths, desc=f"Processing {Path(split_file).stem}"):
        full_path = os.path.join(video_dir, vpath)

        # Output filename: replace / with _ and change extension
        safe_name = vpath.replace("/", "_").replace("\\", "_")
        safe_name = os.path.splitext(safe_name)[0] + ".npz"
        out_path = os.path.join(output_dir, safe_name)

        # Skip if already processed
        if os.path.exists(out_path):
            stats["skipped"] += 1
            stats["success"] += 1
            continue

        if not os.path.exists(full_path):
            stats["failed"] += 1
            failed_videos.append(vpath)
            continue

        try:
            kp = extract_keypoints_from_video(full_path)
            if kp is None:
                stats["failed"] += 1
                failed_videos.append(vpath)
                continue

            # Extract label from path: "Category/N. SignName/MVI_XXXX.MOV"
            parts = vpath.split("/")
            sign_dir = parts[1] if len(parts) > 1 else "unknown"
            # Sign name after the number: "48. Hello" -> "hello"
            sign_name = sign_dir.split(". ", 1)[-1].lower().replace(" ", "")

            # Get integer label
            label_int = -1
            if label_map and sign_name in label_map:
                label_int = label_map[sign_name]

            # Save as .npz
            np.savez_compressed(
                out_path,
                features=keypoints_to_feature_vector(kp),
                hand1_x=kp["hand1_x"],
                hand1_y=kp["hand1_y"],
                hand2_x=kp["hand2_x"],
                hand2_y=kp["hand2_y"],
                pose_x=kp["pose_x"],
                pose_y=kp["pose_y"],
                label=np.array(label_int),
                label_name=np.array(sign_name),
                fps=np.array(kp["fps"]),
                num_frames=np.array(kp["num_frames"]),
                video_path=np.array(vpath),
            )
            stats["success"] += 1

        except Exception as e:
            stats["failed"] += 1
            failed_videos.append(f"{vpath}: {e}")

    stats["failed_videos"] = failed_videos
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe keypoints from INCLUDE-50 videos"
    )
    parser.add_argument("--video-dir", default="data/raw",
                        help="Root directory with downloaded videos")
    parser.add_argument("--output-dir", default="data/keypoints",
                        help="Output directory for .npz files")
    parser.add_argument("--splits-dir", default="data/splits",
                        help="Directory with split .txt files")
    parser.add_argument("--label-map", default="data/label_maps/label_map_include50.json",
                        help="Path to label map JSON")
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split to process")
    args = parser.parse_args()

    print("=" * 60)
    print("MediaPipe Keypoint Extraction for INCLUDE-50")
    print("=" * 60)

    # Load label map
    label_map = None
    if os.path.exists(args.label_map):
        with open(args.label_map) as f:
            label_map = json.load(f)
        print(f"Label map loaded: {len(label_map)} classes")

    # Process each split
    splits_to_process = []
    if args.split in ("train", "all"):
        splits_to_process.append(("include50_train.txt", "train"))
    if args.split in ("val", "all"):
        splits_to_process.append(("include50_val.txt", "val"))
    if args.split in ("test", "all"):
        splits_to_process.append(("include50_test.txt", "test"))

    all_stats = {}
    for split_file, split_name in splits_to_process:
        split_path = os.path.join(args.splits_dir, split_file)
        out_dir = os.path.join(args.output_dir, split_name)

        print(f"\n--- Processing {split_name} split ---")
        t0 = time.time()
        stats = process_split(split_path, args.video_dir, out_dir, label_map)
        elapsed = time.time() - t0

        print(f"  Success: {stats['success']}/{stats['total']}")
        print(f"  Failed:  {stats['failed']}")
        print(f"  Skipped: {stats['skipped']} (already processed)")
        print(f"  Time:    {elapsed:.1f}s")

        if stats["failed_videos"]:
            print(f"  Failed videos ({len(stats['failed_videos'])}):")
            for fv in stats["failed_videos"][:5]:
                print(f"    {fv}")

        all_stats[split_name] = stats

    # Save extraction report
    report_path = os.path.join(args.output_dir, "extraction_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video_dir": args.video_dir,
        "output_dir": args.output_dir,
    }
    for sname, stats in all_stats.items():
        report[sname] = {
            "total": stats["total"],
            "success": stats["success"],
            "failed": stats["failed"],
        }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
