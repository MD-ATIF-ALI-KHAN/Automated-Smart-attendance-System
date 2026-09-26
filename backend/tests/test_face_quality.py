"""
Unit Tests for Face Quality Assessment
Tests blur detection, brightness, size, contrast, and combined scoring
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.face_quality import FaceQualityAssessor


class TestFaceQualityAssessor(unittest.TestCase):
    """Test suite for FaceQualityAssessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.assessor = FaceQualityAssessor()
        
    def test_initialization(self):
        """Test that assessor initializes correctly"""
        self.assertIsNotNone(self.assessor)
        
    def test_blur_detection_sharp_image(self):
        """Test blur detection on a sharp image"""
        # Create a sharp image with high-frequency content
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add checkerboard pattern (high frequency)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i:i+20, j:j+20] = 255
        
        result = self.assessor.assess_quality(image)
        
        # Sharp image should have high blur score
        self.assertGreater(result['blur'], 0.5, "Sharp image should have high blur score")
        
    def test_blur_detection_blurry_image(self):
        """Test blur detection on a blurry image"""
        # Create a blurry image (low frequency)
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Add slight gradient (low frequency)
        for i in range(200):
            image[i, :] = int(128 + i * 0.5)
        
        result = self.assessor.assess_quality(image)
        
        # Blurry image should have low blur score
        self.assertLess(result['blur'], 0.5, "Blurry image should have low blur score")
        
    def test_brightness_assessment_optimal(self):
        """Test brightness assessment on optimally lit image"""
        # Create image with optimal brightness (around 100-150)
        image = np.ones((200, 200, 3), dtype=np.uint8) * 120
        
        result = self.assessor.assess_quality(image)
        
        # Optimal brightness should have high score
        self.assertGreater(result['brightness'], 0.8, "Optimal brightness should have high score")
        
    def test_brightness_assessment_too_dark(self):
        """Test brightness assessment on dark image"""
        # Create very dark image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 20
        
        result = self.assessor.assess_quality(image)
        
        # Dark image should have low brightness score
        self.assertLess(result['brightness'], 0.5, "Dark image should have low brightness score")
        
    def test_brightness_assessment_too_bright(self):
        """Test brightness assessment on overexposed image"""
        # Create very bright image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 240
        
        result = self.assessor.assess_quality(image)
        
        # Overexposed image should have low brightness score
        self.assertLess(result['brightness'], 0.5, "Overexposed image should have low brightness score")
        
    def test_size_checking_adequate(self):
        """Test size checking on adequately sized image"""
        # Create image larger than minimum (80x80)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = self.assessor.assess_quality(image)
        
        # Adequate size should have perfect score
        self.assertEqual(result['size'], 1.0, "Adequate size should have perfect score")
        
    def test_size_checking_too_small(self):
        """Test size checking on small image"""
        # Create image smaller than minimum
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        
        result = self.assessor.assess_quality(image)
        
        # Small image should have low size score
        self.assertLess(result['size'], 1.0, "Small image should have low size score")
        
    def test_contrast_measurement_good(self):
        """Test contrast measurement on high-contrast image"""
        # Create high-contrast image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:100, :] = 255  # Half white
        # Bottom half stays black
        
        result = self.assessor.assess_quality(image)
        
        # High contrast should have high score
        self.assertGreater(result['contrast'], 0.5, "High contrast should have high score")
        
    def test_contrast_measurement_low(self):
        """Test contrast measurement on low-contrast image"""
        # Create low-contrast image (all similar values)
        image = np.ones((200, 200, 3), dtype=np.uint8) * 100
        # Add tiny variation
        image[:100, :] = 105
        
        result = self.assessor.assess_quality(image)
        
        # Low contrast should have low score
        self.assertLess(result['contrast'], 0.5, "Low contrast should have low score")
        
    def test_combined_quality_score(self):
        """Test combined quality scoring"""
        # Create a good quality image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Checkerboard for sharpness
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i:i+20, j:j+20] = 150
                else:
                    image[i:i+20, j:j+20] = 50
        
        result = self.assessor.assess_quality(image)
        
        # Check that overall score is calculated
        self.assertIn('overall', result)
        self.assertGreater(result['overall'], 0.0)
        self.assertLessEqual(result['overall'], 1.0)
        
        # Current implementation exposes a recommendation rather than a boolean passed flag.
        self.assertIn('recommendation', result)
        self.assertIn(result['recommendation'], {'accept', 'warning', 'reject'})
        
    def test_quality_threshold(self):
        """Test quality threshold checking"""
        # Create a poor quality image
        poor_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        result = self.assessor.assess_quality(poor_image)
        
        # Current implementation classifies this quality level through its recommendation.
        self.assertIn(result['recommendation'], {'warning', 'reject'})
        
    def test_issues_list(self):
        """Test that issues are properly identified"""
        # Create image with multiple issues
        # Too small, low contrast, poor brightness
        image = np.ones((50, 50, 3), dtype=np.uint8) * 20
        
        result = self.assessor.assess_quality(image)
        
        # Current implementation reports component scores and a recommendation;
        # it does not expose an issues list.
        self.assertIn('recommendation', result)
        self.assertEqual(result['recommendation'], 'reject')
        
    def test_empty_image(self):
        """Test handling of empty/invalid image"""
        # Create empty image
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        
        result = self.assessor.assess_quality(image)
        
        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertEqual(result['overall'], 0.0)
        
    def test_grayscale_image(self):
        """Test handling of grayscale image"""
        # Create grayscale image
        image = np.ones((200, 200), dtype=np.uint8) * 128
        
        result = self.assessor.assess_quality(image)
        
        # Should handle grayscale images
        self.assertIsNotNone(result)
        self.assertIn('overall', result)


if __name__ == '__main__':
    unittest.main()
