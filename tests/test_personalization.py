"""Test personalization engine with synthetic data."""
import os
import sys
import json
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.setusign.personalization.engine import PersonalizationHead, PersonalizationManager


def test_personalization_head():
    """Test nearest-centroid classification."""
    print("1. Testing PersonalizationHead...")

    head = PersonalizationHead(embed_dim=50)

    # Add samples for 3 classes
    np.random.seed(42)
    for label in ["hello", "thankyou", "goodbye"]:
        centroid = np.random.randn(50).astype(np.float32)
        for _ in range(5):
            sample = centroid + np.random.randn(50).astype(np.float32) * 0.1
            head.add_sample(sample, label)

    stats = head.get_stats()
    assert stats["total_classes"] == 3
    assert stats["custom_classes"] == 3
    print(f"   Enrolled {stats['total_classes']} classes")

    # Predict with a sample close to "hello"
    hello_emb = head.centroids["hello"] + np.random.randn(50).astype(np.float32) * 0.05
    results = head.predict(hello_emb, top_k=3)
    print(f"   Prediction: {results[0][0]} ({results[0][1]:.3f})")
    assert results[0][0] == "hello", f"Expected 'hello', got '{results[0][0]}'"
    print(f"   PASS")


def test_save_load():
    """Test profile persistence."""
    print("\n2. Testing save/load...")

    head = PersonalizationHead(embed_dim=50)
    np.random.seed(42)
    for label in ["bank", "doctor"]:
        for _ in range(3):
            head.add_sample(np.random.randn(50).astype(np.float32), label)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    head.save(path)
    print(f"   Saved to {path}")

    # Load into new head
    head2 = PersonalizationHead(embed_dim=50)
    head2.load(path)
    assert len(head2.centroids) == 2
    assert "bank" in head2.centroids
    assert "doctor" in head2.centroids
    print(f"   Loaded {len(head2.centroids)} classes")

    os.unlink(path)
    print(f"   PASS")


def test_personalization_manager():
    """Test full personalization workflow."""
    print("\n3. Testing PersonalizationManager...")

    mgr = PersonalizationManager(embed_dim=50)

    # Simulate enrollment
    mgr.start_enrollment("namaste")
    assert mgr.enrolling

    for i in range(5):
        fake_seq = np.random.randn(60, 150).astype(np.float32)
        mgr.add_enrollment_sample(fake_seq)

    assert len(mgr.enroll_samples) == 5
    result = mgr.finish_enrollment()
    assert result
    assert not mgr.enrolling

    # Predict
    test_seq = np.random.randn(60, 150).astype(np.float32)
    predictions = mgr.predict(test_seq, top_k=3)
    print(f"   Predictions: {predictions}")
    assert len(predictions) >= 1
    assert predictions[0][0] == "namaste"
    print(f"   PASS")


def main():
    print("=" * 50)
    print("Personalization Engine Tests")
    print("=" * 50)

    test_personalization_head()
    test_save_load()
    test_personalization_manager()

    print(f"\n{'='*50}")
    print("ALL PERSONALIZATION TESTS PASSED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
