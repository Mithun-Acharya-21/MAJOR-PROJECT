"""
Tests for the face pipeline model components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_face_cnn_forward():
    """Test FaceCNN forward pass produces correct shapes."""
    import torch
    from ml.face_pipeline.cnn_model import FaceCNN

    model = FaceCNN(num_classes=2, embed_dim=256)
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224)

    logits, embedding = model(img)

    assert logits.shape == (batch_size, 2)
    assert embedding.shape == (batch_size, 256)


def test_face_cnn_gradcam_compatibility():
    """Test that the backbone supports Grad-CAM hooks."""
    import torch
    from ml.face_pipeline.cnn_model import FaceCNN

    model = FaceCNN(num_classes=2)
    img = torch.randn(1, 3, 224, 224)

    # Hook should be attachable to backbone[-1]
    activations = []
    h = model.backbone[-1].register_forward_hook(
        lambda m, i, o: activations.append(o.detach())
    )

    model(img)
    h.remove()

    assert len(activations) == 1
    assert activations[0].shape[1] == 512  # ResNet-18 layer4 output channels


def test_face_preprocessing():
    """Test face preprocessing produces correct tensor shape."""
    import torch
    from ml.face_pipeline.preprocessing import preprocess_image_bytes

    # Create a dummy 100x100 RGB image
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    tensor = preprocess_image_bytes(img_bytes, target_size=224)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_face_detection():
    """Test face detection on a synthetic image."""
    import numpy as np
    from ml.face_pipeline.face_detection import detect_faces

    # Create a blank image — should still return a fallback bounding box
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    faces = detect_faces(image, method="fallback")

    assert len(faces) >= 1
    assert len(faces[0]) == 4  # (x, y, w, h)


if __name__ == "__main__":
    tests = [
        test_face_cnn_forward,
        test_face_cnn_gradcam_compatibility,
        test_face_preprocessing,
        test_face_detection,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
