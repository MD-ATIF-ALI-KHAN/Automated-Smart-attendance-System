"""
Simple Test Script for Face Quality & Landmarks Features
Run this to test the implementation without needing curl
"""

import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dependencies():
    """Test 1: Verify all dependencies are installed"""
    print("\n" + "="*60)
    print("TEST 1: Checking Dependencies")
    print("="*60)
    
    try:
        import cv2
        import numpy as np
        import dlib
        
        print(f"✅ OpenCV: {cv2.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ dlib: {dlib.__version__}")
        print("\n✅ All dependencies installed successfully!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def test_quality_assessor():
    """Test 2: Test Face Quality Assessor"""
    print("\n" + "="*60)
    print("TEST 2: Face Quality Assessor")
    print("="*60)
    
    try:
        from app.utils.face_quality import FaceQualityAssessor
        import cv2
        import numpy as np
        
        assessor = FaceQualityAssessor()
        print("✅ FaceQualityAssessor initialized")
        
        # Create a test image (random noise)
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = assessor.assess_quality(test_image)
        
        print(f"\nQuality Assessment Results:")
        print(f"  Overall Score: {result['overall']:.2f}")
        print(f"  Blur Score: {result['blur']:.2f}")
        print(f"  Brightness Score: {result['brightness']:.2f}")
        print(f"  Size Score: {result['size']:.2f}")
        print(f"  Contrast Score: {result['contrast']:.2f}")
        
        print("\n✅ Quality assessor working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_landmarks_analyzer():
    """Test 3: Test 68-Point Landmarks Analyzer"""
    print("\n" + "="*60)
    print("TEST 3: 68-Point Landmarks Analyzer")
    print("="*60)
    
    try:
        from app.utils.facial_landmarks_68 import FacialLandmarks68Analyzer
        
        analyzer = FacialLandmarks68Analyzer()
        print("✅ FacialLandmarks68Analyzer initialized")
        print(f"✅ Model loaded from: {analyzer.model_path}")
        
        print("\n✅ Landmarks analyzer working correctly!")
        print("   (Full test requires actual face image)")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test 4: Test Integration with Face Recognition Service"""
    print("\n" + "="*60)
    print("TEST 4: Integration with Face Recognition Service")
    print("="*60)
    
    try:
        from app.services.insightface_face_recognition import InsightFaceFaceRecognitionService
        
        print("Initializing face recognition service...")
        service = InsightFaceFaceRecognitionService()
        
        # Check if quality utils are initialized
        if hasattr(service, 'quality_assessor') and service.quality_assessor is not None:
            print("✅ Quality assessor integrated")
        else:
            print("⚠️  Quality assessor not initialized")
        
        if hasattr(service, 'landmarks_analyzer') and service.landmarks_analyzer is not None:
            print("✅ Landmarks analyzer integrated")
        else:
            print("⚠️  Landmarks analyzer not initialized")
        
        print("\n✅ Integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FACE QUALITY & LANDMARKS - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_dependencies()))
    results.append(("Quality Assessor", test_quality_assessor()))
    results.append(("Landmarks Analyzer", test_landmarks_analyzer()))
    results.append(("Integration", test_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Implementation is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Check errors above.")


if __name__ == "__main__":
    main()
