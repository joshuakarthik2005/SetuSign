"""
Dataset loader for INCLUDE-50 keypoint data.
=============================================
Loads .npz keypoint files, applies augmentations, pads/truncates to
fixed length, and provides PyTorch DataLoader.
"""

import os
import glob
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KeypointsDataset(Dataset):
    """
    PyTorch Dataset for INCLUDE-50 keypoint sequences.
    
    Each sample is a [max_len, num_features] tensor of keypoint coordinates,
    zero-padded or truncated to max_len frames.
    """

    def __init__(self, keypoints_dir, label_map, max_len=200,
                 augment=False, feature_key="features"):
        """
        Args:
            keypoints_dir: Directory containing .npz files
            label_map: Dict mapping sign names to integer labels
            max_len: Maximum sequence length (pad/truncate to this)
            augment: Whether to apply data augmentations
            feature_key: Key in .npz to use as features
        """
        self.files = sorted(glob.glob(os.path.join(keypoints_dir, "*.npz")))
        self.label_map = label_map
        self.max_len = max_len
        self.augment = augment
        self.feature_key = feature_key
        self.num_classes = len(label_map)

        # Pre-filter to only files with valid labels
        valid_files = []
        for f in self.files:
            try:
                data = np.load(f, allow_pickle=True)
                label_name = str(data["label_name"])
                if label_name in label_map:
                    valid_files.append(f)
            except Exception:
                pass
        self.files = valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        features = data[self.feature_key].astype(np.float32)  # [T, F]
        label_name = str(data["label_name"])
        label = self.label_map[label_name]

        # Apply augmentations
        if self.augment:
            features = self._augment(features)

        # Pad or truncate to max_len
        T, F = features.shape
        if T > self.max_len:
            # Uniform downsample
            indices = np.linspace(0, T - 1, self.max_len, dtype=int)
            features = features[indices]
        elif T < self.max_len:
            pad = np.zeros((self.max_len - T, F), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)

        # Create attention mask (1 for real frames, 0 for padding)
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:min(T, self.max_len)] = 1.0

        return {
            "data": torch.from_numpy(features),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "length": min(T, self.max_len),
        }

    def _augment(self, features):
        """Apply random augmentations to keypoint sequence."""
        # 1. Random rotation (small angle)
        if random.random() < 0.4:
            angle = random.uniform(-7, 7) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            T, F = features.shape
            # Apply rotation to x,y pairs
            # Features layout: [h1x(21), h1y(21), h2x(21), h2y(21), px(33), py(33)]
            x_indices = list(range(0, 21)) + list(range(42, 63)) + list(range(84, 117))
            y_indices = list(range(21, 42)) + list(range(63, 84)) + list(range(117, 150))
            for xi, yi in zip(x_indices, y_indices):
                if xi < F and yi < F:
                    x = features[:, xi].copy()
                    y = features[:, yi].copy()
                    features[:, xi] = x * cos_a - y * sin_a
                    features[:, yi] = x * sin_a + y * cos_a

        # 2. Gaussian noise
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.005, features.shape).astype(np.float32)
            # Only add noise to non-zero values (don't add to padding)
            mask = (features != 0).astype(np.float32)
            features = features + noise * mask

        # 3. Random temporal scaling (speed up/slow down)
        if random.random() < 0.4:
            T = features.shape[0]
            scale = random.uniform(0.8, 1.2)
            new_T = max(1, int(T * scale))
            indices = np.linspace(0, T - 1, new_T, dtype=int)
            features = features[indices]

        # 4. Random cutout (zero out random frames)
        if random.random() < 0.3:
            T = features.shape[0]
            cut_len = random.randint(1, max(1, T // 10))
            start = random.randint(0, max(0, T - cut_len))
            features[start:start+cut_len] = 0.0

        return features


def get_dataloaders(keypoints_base_dir, label_map_path, batch_size=32,
                    max_len=200, num_workers=0):
    """
    Create train, val, test DataLoaders.
    
    Args:
        keypoints_base_dir: Base directory containing train/, val/, test/ subdirs
        label_map_path: Path to label_map_include50.json
        batch_size: Batch size
        max_len: Max sequence length
        num_workers: DataLoader workers
    
    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    with open(label_map_path) as f:
        label_map = json.load(f)

    loaders = {}
    for split in ["train", "val", "test"]:
        kp_dir = os.path.join(keypoints_base_dir, split)
        if not os.path.exists(kp_dir):
            print(f"WARNING: {kp_dir} not found, skipping")
            continue

        ds = KeypointsDataset(
            keypoints_dir=kp_dir,
            label_map=label_map,
            max_len=max_len,
            augment=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"  {split}: {len(ds)} samples, {len(loaders[split])} batches")

    return loaders
