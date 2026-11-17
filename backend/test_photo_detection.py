#!/usr/bin/env python3
"""
Test script to verify photo detection is working correctly.
Run this from the project root directory.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.services.photo_service import analyze_photo
from backend.app.main import _state


def test_photo_detection():
    """Test photo detection with a sample image."""
    # Find a test image
    test_img = Path("data/fer2013/train/angry/Training_10118481.jpg")
    
    if not test_img.exists():
        print(f"Test image not found: {test_img}")
        return False
    
    print(f"Testing with image: {test_img}")
    
    # Load models
    print("\nLoading models...")
    _state.load()
    print("Models loaded successfully!")
    
    # Read image
    image_bytes = test_img.read_bytes()
    print(f"Image size: {len(image_bytes)} bytes")
    
    # Run analysis
    print("\nRunning photo analysis...")
    result = analyze_photo(image_bytes, _state)
    
    # Print results
    print("\n" + "="*50)
    print("PHOTO ANALYSIS RESULTS")
    print("="*50)
    print(f"  Prediction:       {result['prediction']}")
    print(f"  Probability:      {result['probability']}%")
    print(f"  Confidence:       {result['confidence']}%")
    print(f"  Risk level:       {result['risk_level']}")
    print(f"  Face detected:    {result['features'].get('face_detected', False)}")
    print(f"  Dominant emotion: {result['features'].get('dominant_emotion', 'N/A')}")
    print(f"  Analysis method:  {result['features'].get('analysis_method', 'N/A')}")
    print(f"  Explanation:      {result['explanation'].get('method', 'N/A')}")
    print(f"  Has GradCAM:      {result['explanation'].get('gradcam') is not None}")
    
    # Verify face detection worked
    if result['features'].get('face_detected'):
        print("\n[OK] Face detection is working correctly!")
        return True
    else:
        print("\n[FAIL] Face detection failed!")
        return False


if __name__ == "__main__":
    success = test_photo_detection()
    sys.exit(0 if success else 1)
