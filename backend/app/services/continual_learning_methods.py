"""
Continual Learning Methods for Multi-Encoding Face Recognition
These methods enable the system to improve recognition over time by adding
high-quality encodings from attendance photos.
"""

import cv2
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


async def add_encoding_from_attendance(
    self,
    student_id: str,
    face_crop: np.ndarray,
    confidence: float,
    quality_score: float,
    source_image_path: str
) -> bool:
    """
    Add a high-quality encoding from an attendance photo to improve recognition
    
    Args:
        student_id: Student ID
        face_crop: Cropped face image from attendance photo
        confidence: Match confidence (0-1)
        quality_score: Face quality score (0-1)
        source_image_path: Path to source attendance image
    
    Returns:
        True if encoding was added, False otherwise
    """
    try:
        # Load continual learning settings
        settings = self._get_continual_learning_settings()
        
        if not settings.get('enabled', False):
            return False
        
        # Check confidence threshold
        min_confidence = settings.get('minConfidenceThreshold', 0.6)
        if confidence < min_confidence:
            logger.info(f"Skipping encoding addition for {student_id}: confidence {confidence:.3f} < {min_confidence}")
            return False
        
        # Check quality threshold
        min_quality = settings.get('minQualityThreshold', 0.7)
        if quality_score < min_quality:
            logger.info(f"Skipping encoding addition for {student_id}: quality {quality_score:.3f} < {min_quality}")
            return False
        
        # Generate encoding from face crop
        # Save face crop temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, face_crop)
        
        try:
            # Generate encoding
            new_encoding = await self.generate_encoding(tmp_path, student_id, is_video_frame=True)
            
            if new_encoding is None:
                logger.warning(f"Failed to generate encoding from attendance photo for {student_id}")
                return False
            
            # Assess encoding quality against existing encodings
            encoding_quality = await self._assess_encoding_quality(new_encoding, student_id)
            
            if encoding_quality < 0.5:
                logger.info(f"Skipping low-quality encoding for {student_id} (quality: {encoding_quality:.3f})")
                return False
            
            # Add to student's encoding collection
            if student_id not in self.known_face_encodings:
                self.known_face_encodings[student_id] = []
            
            encodings = self.known_face_encodings[student_id]
            if not isinstance(encodings, list):
                encodings = [encodings]
                self.known_face_encodings[student_id] = encodings
            
            # Check max encodings limit
            max_encodings = settings.get('maxEncodingsPerStudent', 10)
            if len(encodings) >= max_encodings:
                # Remove oldest encoding (FIFO)
                encodings.pop(0)
                logger.info(f"Removed oldest encoding for {student_id} (limit: {max_encodings})")
            
            # Add new encoding
            encodings.append(new_encoding)
            
            # Save updated encodings
            await self.save_encodings()
            
            # Log addition
            if settings.get('logEncodingAdditions', True):
                logger.info(f"✅ Added encoding from attendance for {student_id}")
                logger.info(f"   Total encodings: {len(encodings)}")
                logger.info(f"   Confidence: {confidence:.3f}, Quality: {quality_score:.3f}")
                logger.info(f"   Encoding quality: {encoding_quality:.3f}")
                logger.info(f"   Source: {source_image_path}")
            
            # Save face crop if enabled
            if settings.get('saveFaceCrops', False):
                await self._save_face_crop(student_id, face_crop, source_image_path)
            
            return True
            
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        logger.error(f"Error adding encoding from attendance for {student_id}: {e}")
        return False


async def _assess_encoding_quality(self, encoding: np.ndarray, student_id: str) -> float:
    """
    Assess quality of new encoding against existing ones
    
    Args:
        encoding: New encoding to assess
        student_id: Student ID
    
    Returns:
        Quality score (0-1), higher is better
    """
    try:
        if student_id not in self.known_face_encodings:
            return 1.0  # First encoding, accept it
        
        existing_encodings = self.known_face_encodings[student_id]
        if not isinstance(existing_encodings, list):
            existing_encodings = [existing_encodings]
        
        if len(existing_encodings) == 0:
            return 1.0
        
        # Calculate similarity with existing encodings
        similarities = []
        for existing in existing_encodings:
            # Cosine similarity
            similarity = np.dot(encoding, existing)
            similarities.append(similarity)
        
        # Average similarity
        avg_similarity = np.mean(similarities)
        
        # High similarity = good quality (consistent with existing)
        # Low similarity = might be poor quality or different pose
        return float(avg_similarity)
        
    except Exception as e:
        logger.error(f"Error assessing encoding quality: {e}")
        return 0.5  # Default to medium quality


async def _save_face_crop(self, student_id: str, face_crop: np.ndarray, source_path: str):
    """Save face crop from attendance for future reference"""
    try:
        import os
        from datetime import datetime
        
        # Create directory for face crops
        crops_dir = f"data/attendance_face_crops/{student_id}"
        os.makedirs(crops_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_filename = f"{timestamp}.jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        
        # Save crop
        cv2.imwrite(crop_path, face_crop)
        logger.info(f"Saved face crop to {crop_path}")
        
    except Exception as e:
        logger.error(f"Error saving face crop: {e}")


def _get_continual_learning_settings(self) -> Dict:
    """Get continual learning settings from app_settings.json"""
    try:
        import json
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                return settings.get('continualLearning', {
                    'enabled': True,
                    'minConfidenceThreshold': 0.6,
                    'minQualityThreshold': 0.7,
                    'maxEncodingsPerStudent': 10,
                    'saveFaceCrops': True,
                    'logEncodingAdditions': True
                })
        return {}
    except Exception as e:
        logger.error(f"Error loading continual learning settings: {e}")
        return {'enabled': False}


# Add these methods to InsightFaceFaceRecognitionService class
YOLOv8FaceRecognitionService.add_encoding_from_attendance = add_encoding_from_attendance
YOLOv8FaceRecognitionService._assess_encoding_quality = _assess_encoding_quality
YOLOv8FaceRecognitionService._save_face_crop = _save_face_crop
YOLOv8FaceRecognitionService._get_continual_learning_settings = _get_continual_learning_settings
