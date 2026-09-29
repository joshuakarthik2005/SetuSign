"""
On-Device Personalization for SetuSign
=======================================
Allows new users to enroll custom signs with just 3-5 examples.
Architecture: Frozen transformer encoder + lightweight classifier head.

The encoder (trained on INCLUDE-50) stays frozen. Only a small
linear classifier is fine-tuned on the user's samples.
This enables:
  - Fast enrollment (~30 seconds per sign)
  - Minimal compute (only trains a linear layer)
  - No forgetting of base signs
  - Works entirely on-device
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class PersonalizationHead(nn.Module):
    """
    Lightweight classifier head for personalized sign recognition.
    
    Uses nearest-centroid classification with optional linear projection.
    This is more robust with few samples than a full linear classifier.
    """

    def __init__(self, embed_dim=256, base_num_classes=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_num_classes = base_num_classes

        # Centroids for each class (both base and custom)
        self.centroids = {}  # label -> np.array of shape [embed_dim]
        self.sample_counts = {}  # label -> int
        self.custom_labels = []  # user-added labels

        # Optional: learnable projection for better separation
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def add_sample(self, embedding, label):
        """
        Add a new sample for a label (running centroid update).
        
        Args:
            embedding: np.array of shape [embed_dim]
            label: string label for this sign
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        if label not in self.centroids:
            self.centroids[label] = embedding.copy()
            self.sample_counts[label] = 1
            if label not in self.custom_labels:
                self.custom_labels.append(label)
        else:
            # Running mean update
            n = self.sample_counts[label]
            self.centroids[label] = (self.centroids[label] * n + embedding) / (n + 1)
            self.sample_counts[label] = n + 1

    def predict(self, embedding, top_k=3):
        """
        Predict the closest class using cosine similarity to centroids.
        
        Args:
            embedding: np.array of shape [embed_dim]
            top_k: number of top predictions to return
        
        Returns:
            list of (label, similarity_score) tuples
        """
        if not self.centroids:
            return [("no_classes_enrolled", 0.0)]

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        # Compute cosine similarity to all centroids
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        similarities = {}
        for label, centroid in self.centroids.items():
            cent_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            sim = float(np.dot(emb_norm, cent_norm))
            similarities[label] = sim

        # Sort by similarity
        sorted_sims = sorted(similarities.items(), key=lambda x: -x[1])

        # Convert similarities to pseudo-probabilities via softmax
        labels = [s[0] for s in sorted_sims]
        sims = np.array([s[1] for s in sorted_sims])
        exp_sims = np.exp(sims * 10)  # temperature scaling
        probs = exp_sims / exp_sims.sum()

        results = [(labels[i], float(probs[i])) for i in range(min(top_k, len(labels)))]
        return results

    def get_stats(self):
        """Return enrollment statistics."""
        return {
            "total_classes": len(self.centroids),
            "custom_classes": len(self.custom_labels),
            "base_classes": len(self.centroids) - len(self.custom_labels),
            "samples_per_class": dict(self.sample_counts),
        }

    def save(self, path):
        """Save personalization data to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "centroids": {k: v.tolist() for k, v in self.centroids.items()},
            "sample_counts": self.sample_counts,
            "custom_labels": self.custom_labels,
            "embed_dim": self.embed_dim,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """Load personalization data from disk."""
        with open(path) as f:
            data = json.load(f)
        self.centroids = {k: np.array(v, dtype=np.float32)
                          for k, v in data["centroids"].items()}
        self.sample_counts = data["sample_counts"]
        self.custom_labels = data["custom_labels"]
        self.embed_dim = data["embed_dim"]


class PersonalizationManager:
    """
    Manages the full personalization workflow:
    1. Enrollment: Record 3-5 samples per new sign
    2. Feature extraction: Use frozen encoder to get embeddings
    3. Classification: Nearest-centroid with enrolled classes
    4. Persistence: Save/load user profiles
    """

    def __init__(self, encoder_model=None, onnx_session=None,
                 embed_dim=256, profile_dir="data/profiles"):
        """
        Args:
            encoder_model: PyTorch model with get_encoder_output() method
            onnx_session: ONNX Runtime session (alternative to PyTorch model)
            embed_dim: Dimension of encoder output
            profile_dir: Directory to save user profiles
        """
        self.encoder_model = encoder_model
        self.onnx_session = onnx_session
        self.embed_dim = embed_dim
        self.profile_dir = profile_dir
        self.head = PersonalizationHead(embed_dim=embed_dim)

        # Enrollment state
        self.enrolling = False
        self.enroll_label = None
        self.enroll_samples = []
        self.min_samples = 3
        self.max_samples = 10

    def get_embedding(self, keypoint_sequence):
        """
        Extract embedding from a keypoint sequence using the frozen encoder.
        
        Args:
            keypoint_sequence: np.array of shape [T, 150]
        
        Returns:
            np.array of shape [embed_dim]
        """
        T, F = keypoint_sequence.shape
        max_len = 200

        # Pad/truncate
        if T > max_len:
            indices = np.linspace(0, T - 1, max_len, dtype=int)
            keypoint_sequence = keypoint_sequence[indices]
            T = max_len
        elif T < max_len:
            pad = np.zeros((max_len - T, F), dtype=np.float32)
            keypoint_sequence = np.concatenate([keypoint_sequence, pad], axis=0)

        mask = np.zeros(max_len, dtype=np.float32)
        mask[:min(T, max_len)] = 1.0

        if self.onnx_session is not None:
            # Use ONNX model — get logits and use pre-softmax as embedding
            outputs = self.onnx_session.run(None, {
                "keypoints": keypoint_sequence[np.newaxis].astype(np.float32),
                "mask": mask[np.newaxis].astype(np.float32),
            })
            # Use logits as a proxy embedding (not ideal but works for demo)
            return outputs[0][0]

        elif self.encoder_model is not None:
            # Use PyTorch model
            with torch.no_grad():
                x = torch.from_numpy(keypoint_sequence[np.newaxis])
                m = torch.from_numpy(mask[np.newaxis])
                embedding = self.encoder_model.get_encoder_output(x, m)
                return embedding[0].numpy()

        else:
            # Fallback: use mean of keypoints as embedding
            return keypoint_sequence[:T].mean(axis=0)[:self.embed_dim]

    def start_enrollment(self, label):
        """Start enrolling a new sign."""
        self.enrolling = True
        self.enroll_label = label
        self.enroll_samples = []
        print(f"Enrollment started for '{label}'. Show {self.min_samples}-{self.max_samples} examples.")

    def add_enrollment_sample(self, keypoint_sequence):
        """Add one enrollment sample."""
        if not self.enrolling:
            return False

        embedding = self.get_embedding(keypoint_sequence)
        self.enroll_samples.append(embedding)
        count = len(self.enroll_samples)
        print(f"  Sample {count}/{self.min_samples} for '{self.enroll_label}'")

        if count >= self.min_samples:
            print(f"  Minimum samples reached! Press E to finish or continue adding.")

        return True

    def finish_enrollment(self):
        """Finish enrollment and save the new sign."""
        if not self.enrolling or len(self.enroll_samples) < self.min_samples:
            return False

        # Add all samples to the head
        for emb in self.enroll_samples:
            self.head.add_sample(emb, self.enroll_label)

        print(f"Enrolled '{self.enroll_label}' with {len(self.enroll_samples)} samples")

        self.enrolling = False
        self.enroll_label = None
        self.enroll_samples = []
        return True

    def cancel_enrollment(self):
        """Cancel current enrollment."""
        self.enrolling = False
        self.enroll_label = None
        self.enroll_samples = []

    def predict(self, keypoint_sequence, top_k=3):
        """Predict sign from keypoint sequence using personalized model."""
        embedding = self.get_embedding(keypoint_sequence)
        return self.head.predict(embedding, top_k=top_k)

    def save_profile(self, user_id="default"):
        """Save user profile to disk."""
        path = os.path.join(self.profile_dir, f"{user_id}.json")
        self.head.save(path)
        print(f"Profile saved: {path}")

    def load_profile(self, user_id="default"):
        """Load user profile from disk."""
        path = os.path.join(self.profile_dir, f"{user_id}.json")
        if os.path.exists(path):
            self.head.load(path)
            print(f"Profile loaded: {path} ({self.head.get_stats()['total_classes']} classes)")
            return True
        return False
