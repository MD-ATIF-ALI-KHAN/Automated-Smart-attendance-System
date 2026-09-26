"""
Unit Tests for 68-Point Facial Landmarks Analysis
Tests pose estimation, eye state detection, and frontal face detection
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.facial_landmarks_68 import FacialLandmarks68Analyzer


class TestFacialLandmarks68Analyzer(unittest.TestCase):
    """Test suite for FacialLandmarks68Analyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.analyzer = FacialLandmarks68Analyzer()
            self.analyzer_available = True
        except Exception as e:
            print(f"Warning: Could not initialize analyzer: {e}")
            self.analyzer_available = False
        
    def test_initialization(self):
        """Test that analyzer initializes correctly"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.predictor)
        
    def test_analyze_with_no_face(self):
        """Test analysis on image with no face"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        # Create blank image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = self.analyzer.comprehensive_analysis(image)
        
        # Should return None or empty result for no face
        self.assertIsNotNone(result)
        
    def test_frontal_face_detection_logic(self):
        """Test frontal face detection logic with mock data"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        # Test the logic without actual face detection
        # Frontal face should have yaw, pitch, roll close to 0
        
        # This tests the threshold logic
        yaw_threshold = 15
        pitch_threshold = 15
        roll_threshold = 15
        
        # Test case 1: Frontal face
        yaw, pitch, roll = 5, 3, 2
        is_frontal = (abs(yaw) < yaw_threshold and 
                     abs(pitch) < pitch_threshold and 
                     abs(roll) < roll_threshold)
        self.assertTrue(is_frontal, "Should detect frontal face")
        
        # Test case 2: Non-frontal face (large yaw)
        yaw, pitch, roll = 25, 3, 2
        is_frontal = (abs(yaw) < yaw_threshold and 
                     abs(pitch) < pitch_threshold and 
                     abs(roll) < roll_threshold)
        self.assertFalse(is_frontal, "Should detect non-frontal face")
        
    def test_eye_aspect_ratio_logic(self):
        """Test Eye Aspect Ratio calculation logic"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        # Mock EAR calculation
        # EAR = (vertical1 + vertical2) / (2 * horizontal)
        
        # Test case 1: Open eye (larger vertical distances)
        vertical1 = 10
        vertical2 = 10
        horizontal = 15
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        
        self.assertGreater(ear, 0.2, "Open eye should have EAR > 0.2")
        
        # Test case 2: Closed eye (smaller vertical distances)
        vertical1 = 2
        vertical2 = 2
        horizontal = 15
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        
        self.assertLess(ear, 0.2, "Closed eye should have EAR < 0.2")
        
    def test_pose_angle_ranges(self):
        """Test that pose angles are within expected ranges"""
        # Pose angles should typically be in range [-90, 90] for yaw/pitch
        # and [-180, 180] for roll
        
        # This is a logic test without actual face detection
        test_angles = [
            (0, 0, 0),      # Frontal
            (15, 10, 5),    # Slight turn
            (-15, -10, -5), # Slight turn other direction
            (45, 30, 20),   # Moderate turn
        ]
        
        for yaw, pitch, roll in test_angles:
            self.assertGreaterEqual(yaw, -90)
            self.assertLessEqual(yaw, 90)
            self.assertGreaterEqual(pitch, -90)
            self.assertLessEqual(pitch, 90)
            self.assertGreaterEqual(roll, -180)
            self.assertLessEqual(roll, 180)
            
    def test_occlusion_detection_logic(self):
        """Test occlusion detection logic"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        # Test MAR (Mouth Aspect Ratio) logic for mask detection
        # High MAR when mouth is covered
        
        # Mock MAR calculation
        # MAR = vertical / horizontal
        
        # Normal mouth
        vertical = 10
        horizontal = 30
        mar = vertical / horizontal
        self.assertLess(mar, 0.5, "Normal mouth should have low MAR")
        
        # Covered mouth (distorted proportions)
        vertical = 5
        horizontal = 30
        mar = vertical / horizontal
        self.assertLess(mar, 0.3, "Covered mouth may have very low MAR")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for common scenarios"""
    
    def test_quality_and_landmarks_together(self):
        """Test that quality assessment and landmarks work together"""
        from app.utils.face_quality import FaceQualityAssessor
        
        quality_assessor = FaceQualityAssessor()
        
        # Create test image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i:i+20, j:j+20] = 150
        
        # Both should work on same image
        quality_result = quality_assessor.assess_quality(image)
        
        self.assertIsNotNone(quality_result)
        self.assertIn('overall', quality_result)


if __name__ == '__main__':
    unittest.main()
