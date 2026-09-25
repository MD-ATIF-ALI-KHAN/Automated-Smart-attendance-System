# Try to import face_recognition (optional for cloud deployment)
try:
    from app.utils import face_recognition_wrapper as face_recognition
    FACE_RECOGNITION_AVAILABLE = face_recognition.AVAILABLE
except ImportError:
    face_recognition = None  # type: ignore
    FACE_RECOGNITION_AVAILABLE = False
    
if not FACE_RECOGNITION_AVAILABLE:
    print("⚠️ face_recognition not available (requires dlib compilation)")
    print("   Using InsightFace for face detection only")

import cv2
import numpy as np
import os
import pickle
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from app.services.student_management import StudentManagementService
from app.utils.gpu_image_processing import gpu_processor
from app.utils.gpu_utils import gpu_manager

# Try to import YOLOv8 service
try:
    from app.services.insightface_face_recognition import insightface_face_service
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    insightface_face_service = None  # Define as None if import fails
    INSIGHTFACE_AVAILABLE = False
    print("YOLOv8 not available. Install required packages: ultralytics, deepface, torch")

class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = {}
        self.known_face_names = {}
        # Determine correct path based on working directory
        if os.path.exists("app/data"):
            self.encodings_file = "app/data/encodings/face_encodings.pkl"
        else:
            self.encodings_file = "data/encodings/face_encodings.pkl"
        self.student_service = StudentManagementService()
        
        # Load settings from app_settings.json
        if os.path.exists("app/data"):
            self.settings_file = "app/data/app_settings.json"
        else:
            self.settings_file = "data/app_settings.json"
        
        # YOLOv8 integration
        self.use_insightface = False  # Will be set from settings
        self.insightface_service = insightface_face_service if INSIGHTFACE_AVAILABLE else None
        
        self._load_settings()
        
        # Face recognition parameters - BALANCED for real-world accuracy
        self.face_recognition_tolerance = 0.50  # Balanced tolerance
        self.min_confidence_threshold = 0.45  # Balanced confidence (45%)
        self.strict_mode = False  # Lenient mode for better detection
        self.use_ambiguous_detection = False  # Disable ambiguous detection
        self.face_detection_model = 'hog'  # Default to HOG for speed
        self.num_jitters = 5  # Balanced jitters for attendance
        self.training_jitters = 10  # HIGH quality for training encodings
        self.training_model = 'large'  # Use large model for training
        self.face_detection_upsamples = 0  # No upsampling by default for speed
        
        # Detection mode settings
        self.detection_scales = [1.0, 1.5]  # Reduced scales for speed
        self.use_cnn_for_large = False  # Disable CNN by default
        
        # GPU acceleration setup
        self.gpu_processor = gpu_processor
        self.gpu_manager = gpu_manager
        self._initialize_gpu_acceleration()
        
        # OPTIMIZATION: Add caching
        self._class_students_cache = {}
        self._class_encodings_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._last_cache_update = {}
    
    def _load_settings(self):
        """Load detection and recognition settings from app_settings.json"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    
                    # Check if YOLOv8 should be used
                    self.use_insightface = settings.get('useYOLOv8', True)  # Default to YOLOv8 for better accuracy
                    
                    if self.use_insightface and not INSIGHTFACE_AVAILABLE:
                        print("⚠️  YOLOv8 requested but not available. Falling back to face_recognition library.")
                        self.use_insightface = False
                    
                    print(f"🔧 Recognition System: {'InsightFace (ArcFace)' if self.use_insightface else 'face_recognition (legacy)'}")
                    
                    # Load face detection settings
                    face_detection = settings.get('faceDetection', {})
                    
                    # Apply detection mode settings
                    mode = face_detection.get('mode', 'balanced')
                    
                    if mode == 'fast':
                        # Fast mode: Speed over accuracy
                        self.face_detection_upsamples = 0
                        self.detection_scales = [1.0]
                        self.use_cnn_for_large = False
                        self.num_jitters = 1
                    elif mode == 'accurate':
                        # Accurate mode: Accuracy over speed
                        self.face_detection_upsamples = 1
                        self.detection_scales = [1.0, 1.5, 2.0]
                        self.use_cnn_for_large = True
                        self.num_jitters = 5
                    else:
                        # Balanced mode (default)
                        self.face_detection_upsamples = face_detection.get('upsampleTimes', 0)
                        self.detection_scales = face_detection.get('detectionScales', [1.0, 1.5])
                        self.use_cnn_for_large = face_detection.get('useCNN', False)
                        self.num_jitters = face_detection.get('numJitters', 3)
                    
                    # Load face recognition accuracy settings
                    face_recognition_settings = settings.get('faceRecognition', {})
                    self.face_recognition_tolerance = face_recognition_settings.get('tolerance', 0.50)
                    self.min_confidence_threshold = face_recognition_settings.get('minConfidence', 0.45)
                    self.strict_mode = face_recognition_settings.get('strictMode', False)
                    self.use_ambiguous_detection = face_recognition_settings.get('useAmbiguousDetection', False)
                    self.training_jitters = face_recognition_settings.get('trainingJitters', 10)
                    self.training_model = face_recognition_settings.get('trainingModel', 'large')
                    
                    if not self.use_insightface:
                        print(f"Face Detection Mode: {mode}")
                        print(f"Upsample times: {self.face_detection_upsamples}")
                        print(f"Detection scales: {self.detection_scales}")
                        print(f"Use CNN: {self.use_cnn_for_large}")
                        print(f"Attendance Jitters: {self.num_jitters}")
                        print(f"Training Jitters: {self.training_jitters}")
                        print(f"Training Model: {self.training_model}")
                        print(f"Recognition Tolerance: {self.face_recognition_tolerance}")
                        print(f"Min Confidence: {self.min_confidence_threshold}")
                        print(f"Strict Mode: {self.strict_mode}")
                        print(f"Ambiguous Detection: {self.use_ambiguous_detection}")
        except Exception as e:
            print(f"Error loading settings, using defaults: {e}")
    
    def _initialize_gpu_acceleration(self):
        """Initialize optimized processing - prioritize speed over GPU overhead"""
        try:
            # Minimal logging for faster startup
            gpu_status = self.gpu_manager.get_status()
            print(f"Face Recognition: GPU={gpu_status['gpu_available']}, Model={self.face_detection_model}")
            
        except Exception as e:
            pass  # Silently ignore initialization errors
        
    async def load_encodings(self):
        """Load existing face encodings from file"""
        try:
            # Use YOLOv8 if enabled
            if self.use_insightface and self.yolov8_service:
                await self.yolov8_service.load_encodings()
                print(f"✅ Loaded YOLOv8 encodings for {len(self.yolov8_service.known_face_encodings)} students")
                return
            
            # Legacy system
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', {})
                    self.known_face_names = data.get('names', {})
                print(f"Loaded encodings for {len(self.known_face_encodings)} students")
            else:
                print("No existing encodings found. Starting fresh.")
        except Exception as e:
            print(f"Error loading encodings: {e}")
            self.known_face_encodings = {}
            self.known_face_names = {}
    
    async def save_encodings(self):
        """Save face encodings to file"""
        try:
            os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print("Face encodings saved successfully")
        except Exception as e:
            print(f"Error saving encodings: {e}")
    
    async def generate_encoding(self, image_path: str, student_id: str) -> Optional[np.ndarray]:
        """Generate HIGH-QUALITY face encoding for a student image - GPU OPTIMIZED"""
        try:
            # Check if face_recognition is available
            if not FACE_RECOGNITION_AVAILABLE:
                print("⚠️ face_recognition not available, cannot generate encoding")
                return None
            
            # Use YOLOv8 if enabled
            if self.use_insightface and self.yolov8_service:
                return await self.yolov8_service.generate_encoding(image_path, student_id)
            
            # Legacy system below
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # ALWAYS enhance images for better encoding quality (GPU-accelerated)
            height, width = image.shape[:2]
            image_size = height * width
            
            # Force GPU usage for large images
            if image_size > 500000 and self.gpu_manager.gpu_available:
                # Resize to optimal size for GPU processing if too large
                if image_size > 4000000:  # 4MP
                    scale = (2000000 / image_size) ** 0.5
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = self.gpu_processor.resize_gpu(image, (new_width, new_height))
                
                # GPU-accelerated enhancement
                enhanced_image = self.gpu_processor.enhance_image_gpu(image)
            else:
                # CPU enhancement for smaller images
                enhanced_image = self.enhance_image_quality(image)
            
            # Use BETTER face detection for training
            face_locations = face_recognition.face_locations(  # type: ignore
                enhanced_image, 
                number_of_times_to_upsample=1,  # Upsample for better training detection
                model='cnn' if height * width > 500000 else 'hog'  # Use CNN for medium+ images
            )
            
            if len(face_locations) == 0:
                print(f"⚠️  No faces found in {image_path}")
                return None
            
            if len(face_locations) > 1:
                print(f"⚠️  Multiple faces found in {image_path}. Using the largest face.")
                # Choose the largest face
                face_areas = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
                largest_face_idx = np.argmax(face_areas)
                face_locations = [face_locations[largest_face_idx]]
            
            # Generate HIGH-QUALITY encoding with MAXIMUM jitters and LARGE model
            print(f"Generating HIGH-QUALITY encoding: {self.training_jitters} jitters, model={self.training_model}")
            face_encodings = face_recognition.face_encodings(  # type: ignore
                enhanced_image, 
                face_locations, 
                num_jitters=self.training_jitters,  # HIGH quality - 10 jitters
                model=self.training_model  # Use LARGE model for maximum accuracy
            )
            
            if len(face_encodings) > 0:
                encoding = face_encodings[0]
                
                # Store the encoding (support multiple encodings per student)
                if student_id not in self.known_face_encodings:
                    self.known_face_encodings[student_id] = []
                
                # Add encoding to the list
                self.known_face_encodings[student_id].append(encoding)
                
                # Get student info
                student = await self.student_service.get_student_by_id(student_id)
                if student:
                    self.known_face_names[student_id] = student['name']
                
                print(f"✓ Generated HIGH-QUALITY encoding for {student_id} (total: {len(self.known_face_encodings[student_id])})")
                
                # Save encodings
                await self.save_encodings()
                
                return encoding
            
            return None
            
        except Exception as e:
            print(f"Error generating encoding for {student_id}: {e}")
            return None
    
    async def clear_all_encodings(self) -> bool:
        """Clear all face encodings"""
        try:
            # Use YOLOv8 if enabled
            if self.use_insightface and self.yolov8_service:
                return await self.yolov8_service.clear_all_encodings()
            
            # Legacy system
            # Clear in-memory encodings
            self.known_face_encodings = {}
            self.known_face_names = {}
            
            # Remove encodings file
            if os.path.exists(self.encodings_file):
                os.remove(self.encodings_file)
            
            print("Cleared all face encodings")
            return True
            
        except Exception as e:
            print(f"Error clearing all encodings: {e}")
            return False
    
    async def remove_encoding(self, student_id: str):
        """Remove a student's face encoding"""
        try:
            # Use YOLOv8 if enabled
            if self.use_insightface and self.yolov8_service:
                await self.yolov8_service.remove_encoding(student_id)
                return
            
            # Legacy system
            if student_id in self.known_face_encodings:
                del self.known_face_encodings[student_id]
            if student_id in self.known_face_names:
                del self.known_face_names[student_id]
            
            await self.save_encodings()
            print(f"Removed encoding for student {student_id}")
        except Exception as e:
            print(f"Error removing encoding for {student_id}: {e}")
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """GPU-accelerated image quality enhancement for better face detection"""
        try:
            # Use GPU-accelerated image processing
            enhanced_image = self.gpu_processor.enhance_image_gpu(image)
            print(f"Image enhanced using {'GPU' if self.gpu_manager.gpu_available else 'CPU'} acceleration")
            return enhanced_image
        except Exception as e:
            print(f"GPU enhancement failed, falling back to CPU: {e}")
            return self._enhance_image_cpu_fallback(image)
    
    def _enhance_image_cpu_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced CPU image processing for long-distance face recognition
        Optimized for classroom scenarios with distant students
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Step 1: Denoise first for better processing
        image_rgb = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)
        
        # Step 2: Enhanced CLAHE for better contrast (critical for distant faces)
        if len(image_rgb.shape) == 3:
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # More aggressive CLAHE for distant faces
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            image_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            image_rgb = clahe.apply(image_rgb)
        
        # Step 3: Adaptive histogram equalization for better lighting
        if len(image_rgb.shape) == 3:
            # Apply to each channel
            for i in range(3):
                image_rgb[:, :, i] = cv2.equalizeHist(image_rgb[:, :, i])
        
        # Step 4: Enhanced sharpening for distant faces
        # Use unsharp masking for better edge detection
        gaussian = cv2.GaussianBlur(image_rgb, (0, 0), 2.0)
        image_rgb = cv2.addWeighted(image_rgb, 1.5, gaussian, -0.5, 0)
        
        # Step 5: Additional sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image_rgb, -1, kernel)
        image_rgb = cv2.addWeighted(image_rgb, 0.6, sharpened, 0.4, 0)
        
        # Step 6: Brightness and contrast adjustment for classroom lighting
        alpha = 1.3  # Contrast control (1.0-3.0)
        beta = 20    # Brightness control (0-100)
        image_rgb = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
        
        # Step 7: Gamma correction for better face visibility
        gamma = 1.3  # Increased for better visibility of distant faces
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image_rgb = cv2.LUT(image_rgb, table)
        
        print("Enhanced image for long-distance face detection")
        return image_rgb
    
    def detect_faces_multiple_scales(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """GPU-optimized face detection using multiple scales"""
        try:
            # Use GPU-optimized face detection
            face_locations = self.gpu_processor.detect_faces_gpu_optimized(image)
            print(f"Detected {len(face_locations)} faces using GPU-optimized detection")
            return face_locations
        except Exception as e:
            print(f"GPU face detection failed, using CPU fallback: {e}")
            return self._detect_faces_cpu_fallback(image)
    
    def _detect_faces_cpu_fallback(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """CPU fallback for face detection"""
        face_locations = []
        scales = [1.0, 0.8, 0.6]
        
        for scale in scales:
            if scale != 1.0:
                height, width = image.shape[:2]
                new_height, new_width = int(height * scale), int(width * scale)
                scaled_image = cv2.resize(image, (new_width, new_height))
            else:
                scaled_image = image
            
            locations = face_recognition.face_locations(scaled_image, model=self.face_detection_model)
            
            for (top, right, bottom, left) in locations:
                scaled_location = (
                    int(top / scale),
                    int(right / scale),
                    int(bottom / scale),
                    int(left / scale)
                )
                
                is_duplicate = False
                for existing_location in face_locations:
                    if self._is_same_face(scaled_location, existing_location):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    face_locations.append(scaled_location)
        
        return face_locations
    
    def _is_same_face(self, loc1: Tuple[int, int, int, int], loc2: Tuple[int, int, int, int], threshold: int = 50) -> bool:
        """Check if two face locations represent the same face"""
        top1, right1, bottom1, left1 = loc1
        top2, right2, bottom2, left2 = loc2
        
        # Calculate center points
        center1 = ((left1 + right1) // 2, (top1 + bottom1) // 2)
        center2 = ((left2 + right2) // 2, (top2 + bottom2) // 2)
        
        # Calculate distance between centers
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        
        return distance < threshold
    
    def _detect_faces_long_distance(self, image: np.ndarray, image_size: int) -> List[Tuple[int, int, int, int]]:
        """
        Optimized face detection for long-distance classroom photography
        Uses configurable multi-scale detection based on settings
        """
        try:
            face_locations = []
            
            # Method 1: Optional CNN with upsampling (only if enabled in settings)
            if self.use_cnn_for_large and image_size > 1000000:  # 1MP+
                print("Using CNN model for large classroom image...")
                try:
                    cnn_locations = face_recognition.face_locations(
                        image,
                        number_of_times_to_upsample=self.face_detection_upsamples,
                        model='cnn'
                    )
                    face_locations.extend(cnn_locations)
                    print(f"CNN detection found {len(cnn_locations)} faces")
                except Exception as e:
                    print(f"CNN detection failed: {e}, falling back to HOG")
            
            # Method 2: Multi-scale HOG detection (fast and efficient)
            print(f"Using multi-scale HOG detection with scales: {self.detection_scales}")
            
            for scale_factor in self.detection_scales:
                try:
                    # Only resize if scale > 1.0
                    if scale_factor > 1.0:
                        height, width = image.shape[:2]
                        new_height = int(height * scale_factor)
                        new_width = int(width * scale_factor)
                        scaled_image = cv2.resize(image, (new_width, new_height), 
                                                 interpolation=cv2.INTER_CUBIC)
                    else:
                        scaled_image = image
                    
                    # Detect faces - NO additional upsampling on scaled images for speed
                    hog_locations = face_recognition.face_locations(
                        scaled_image,
                        number_of_times_to_upsample=0 if scale_factor > 1.0 else self.face_detection_upsamples,
                        model='hog'
                    )
                    
                    # Scale back to original coordinates
                    for (top, right, bottom, left) in hog_locations:
                        if scale_factor != 1.0:
                            scaled_location = (
                                int(top / scale_factor),
                                int(right / scale_factor),
                                int(bottom / scale_factor),
                                int(left / scale_factor)
                            )
                        else:
                            scaled_location = (top, right, bottom, left)
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing_location in face_locations:
                            if self._is_same_face(scaled_location, existing_location, threshold=30):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            face_locations.append(scaled_location)
                    
                    print(f"Scale {scale_factor}x found {len(hog_locations)} faces")
                
                except Exception as e:
                    print(f"HOG detection at scale {scale_factor} failed: {e}")
                    continue
            
            # Remove duplicates one more time
            unique_locations = []
            for loc in face_locations:
                is_duplicate = False
                for existing in unique_locations:
                    if self._is_same_face(loc, existing, threshold=30):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_locations.append(loc)
            
            print(f"Total unique faces detected: {len(unique_locations)}")
            return unique_locations
            
        except Exception as e:
            print(f"Long-distance detection failed: {e}, using fallback")
            # Fallback to basic detection
            return face_recognition.face_locations(
                image,
                number_of_times_to_upsample=0,
                model='hog'
            )
    
    async def process_attendance_image(self, image_path: str, class_name: str) -> Dict:
        """Process a classroom image and identify students - OPTIMIZED FOR LONG DISTANCE"""
        try:
            # Check if face_recognition is available
            if not FACE_RECOGNITION_AVAILABLE:
                print("⚠️ face_recognition not available - using YOLOv8 detection only")
                return await self._process_with_yolov8_only(image_path, class_name)
            
            # Use InsightFace if enabled (RECOMMENDED for better accuracy)
            if self.use_insightface and self.yolov8_service:
                print("🚀 Using InsightFace (ArcFace) for attendance processing")
                return await self.yolov8_service.process_attendance_image(image_path, class_name)
            
            # Legacy system below
            print("⚠️  Using legacy face_recognition library (YOLOv8 recommended for better accuracy)")
            
            # Load and enhance the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # OPTIMIZATION 1: ALWAYS enhance for classroom images (better long-distance detection)
            height, width = image.shape[:2]
            image_size = height * width
            
            # Always enhance classroom images for better face detection at distance
            enhanced_image = self.enhance_image_quality(image)
            
            # OPTIMIZATION 2: Multi-scale detection for long-distance faces
            # Use CNN model for better accuracy with distant faces
            face_locations = self._detect_faces_long_distance(enhanced_image, image_size)
            
            print(f"Detected {len(face_locations)} faces in the image")
            
            if len(face_locations) == 0:
                # Get all students in the class - OPTIMIZATION: Cache this
                class_students = await self._get_cached_class_students(class_name)
                return {
                    'present': [],
                    'absent': [{'student_id': s['student_id'], 'name': s['name']} for s in class_students],
                    'total_faces_detected': 0
                }
            
            # OPTIMIZATION 3: Enhanced face encoding for long-distance faces
            # Use configurable jitters based on settings
            face_encodings = face_recognition.face_encodings(
                enhanced_image, 
                face_locations, 
                num_jitters=self.num_jitters,  # Configurable jitters
                model='large'  # Large model for better accuracy with distant faces
            )
            
            # OPTIMIZATION 4: Pre-load class encodings once
            class_encodings_data = await self._get_class_encodings(class_name)
            all_class_encodings = class_encodings_data['encodings']
            student_id_mapping = class_encodings_data['mapping']
            student_info_cache = class_encodings_data['student_info']
            
            if not all_class_encodings:
                # No encodings available for this class
                class_students = await self._get_cached_class_students(class_name)
                return {
                    'present': [],
                    'absent': [{'student_id': s['student_id'], 'name': s['name']} for s in class_students],
                    'total_faces_detected': len(face_locations)
                }
            
            # OPTIMIZATION 5: Batch face comparison for all faces at once
            present_students = []
            identified_student_ids = set()
            
            # Process all face encodings in batch
            for i, face_encoding in enumerate(face_encodings):
                # OPTIMIZATION: Use vectorized operations
                face_distances = face_recognition.face_distance(
                    all_class_encodings, 
                    face_encoding
                )
                
                # Find best match with improved accuracy
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    
                    # Calculate confidence
                    confidence = float(1 - best_distance)
                    
                    # IMPROVED: Stricter matching criteria to prevent false positives
                    # Check both tolerance and minimum confidence
                    is_valid_match = False
                    
                    if self.strict_mode:
                        # Strict mode: Both conditions must be met
                        is_valid_match = (
                            best_distance < self.face_recognition_tolerance and 
                            confidence >= self.min_confidence_threshold
                        )
                    else:
                        # Lenient mode: Either condition can be met
                        is_valid_match = (
                            best_distance < self.face_recognition_tolerance or 
                            confidence >= self.min_confidence_threshold
                        )
                    
                    # Optional: Check for ambiguous matches (only if enabled)
                    if is_valid_match and self.use_ambiguous_detection and len(face_distances) > 1:
                        sorted_indices = np.argsort(face_distances)
                        second_best_distance = face_distances[sorted_indices[1]]
                        distance_gap = second_best_distance - best_distance
                        
                        # Require at least 0.08 difference (relaxed from 0.1)
                        if distance_gap < 0.08:
                            print(f"⚠️  Ambiguous match rejected - best: {best_distance:.3f}, second: {second_best_distance:.3f}, gap: {distance_gap:.3f}")
                            is_valid_match = False
                    
                    if is_valid_match:
                        student_id = student_id_mapping[best_match_index]
                        
                        # Avoid duplicates
                        if student_id not in identified_student_ids:
                            # Use cached student info
                            student_info = student_info_cache.get(student_id)
                            if student_info:
                                present_students.append({
                                    'student_id': student_id,
                                    'name': student_info['name'],
                                    'confidence': float(confidence)  # Convert numpy.float32 to Python float
                                })
                                identified_student_ids.add(student_id)
                                print(f"✓ Identified: {student_info['name']} (confidence: {confidence:.3f}, distance: {best_distance:.3f})")
                    else:
                        print(f"✗ Match rejected - distance: {best_distance:.3f}, confidence: {confidence:.3f}")
            
            # OPTIMIZATION 6: Get absent students from cache
            all_class_students = await self._get_cached_class_students(class_name)
            absent_students = []
            
            for student in all_class_students:
                if student['student_id'] not in identified_student_ids:
                    absent_students.append({
                        'student_id': student['student_id'],
                        'name': student['name']
                    })
            
            # OPTIMIZATION 7: Async save annotated image (don't block)
            import asyncio
            asyncio.create_task(self._save_annotated_image_async(image_path, enhanced_image, face_locations, present_students))
            
            return {
                'present': present_students,
                'absent': absent_students,
                'total_faces_detected': len(face_locations)
            }
            
        except Exception as e:
            print(f"Error processing attendance image: {e}")
            raise e
    
    
    async def update_recognition_parameters(self, tolerance: Optional[float] = None, model: Optional[str] = None):
        """Update face recognition parameters"""
        if tolerance is not None:
            self.face_recognition_tolerance = tolerance
        if model is not None and model in ['hog', 'cnn']:
            self.face_detection_model = model
        
        print(f"Updated parameters: tolerance={self.face_recognition_tolerance}, model={self.face_detection_model}")
    
    async def get_recognition_stats(self) -> Dict:
        """Get face recognition statistics"""
        # Use InsightFace if enabled
        if self.use_insightface and self.yolov8_service:
            stats = await self.yolov8_service.get_recognition_stats()
            stats['system'] = 'InsightFace (ArcFace)'
            return stats
        
        # Legacy system
        return {
            'system': 'face_recognition (legacy)',
            'total_students_enrolled': len(self.known_face_encodings),
            'recognition_tolerance': self.face_recognition_tolerance,
            'detection_model': self.face_detection_model,
            'encodings_file_size': os.path.getsize(self.encodings_file) if os.path.exists(self.encodings_file) else 0
        }
    
    # OPTIMIZATION METHODS - New caching and performance improvements
    
    
    async def _get_cached_class_students(self, class_name: str) -> List[Dict]:
        """Get class students with caching"""
        current_time = datetime.now().timestamp()
        
        # Check if cache is valid
        if (class_name in self._class_students_cache and 
            class_name in self._last_cache_update and
            current_time - self._last_cache_update[class_name] < self._cache_ttl):
            return self._class_students_cache[class_name]
        
        # Load from database and cache
        students = await self.student_service.get_students(class_name)
        self._class_students_cache[class_name] = students
        self._last_cache_update[class_name] = current_time
        
        return students
    
    async def _get_class_encodings(self, class_name: str) -> Dict:
        """Get class encodings with caching"""
        current_time = datetime.now().timestamp()
        
        # Check if cache is valid
        if (class_name in self._class_encodings_cache and 
            class_name in self._last_cache_update and
            current_time - self._last_cache_update[class_name] < self._cache_ttl):
            return self._class_encodings_cache[class_name]
        
        # Load class students
        class_students = await self._get_cached_class_students(class_name)
        
        # Build encodings data
        all_class_encodings = []
        student_id_mapping = []
        student_info_cache = {}
        
        for student in class_students:
            student_id = student['student_id']
            student_info_cache[student_id] = {
                'name': student['name'],
                'class_name': student['class_name']
            }
            
            if student_id in self.known_face_encodings:
                student_encodings = self.known_face_encodings[student_id]
                # Handle both single encoding (legacy) and multiple encodings
                if isinstance(student_encodings, list):
                    all_class_encodings.extend(student_encodings)
                    student_id_mapping.extend([student_id] * len(student_encodings))
                else:
                    all_class_encodings.append(student_encodings)
                    student_id_mapping.append(student_id)
        
        result = {
            'encodings': all_class_encodings,
            'mapping': student_id_mapping,
            'student_info': student_info_cache
        }
        
        # Cache the result
        self._class_encodings_cache[class_name] = result
        self._last_cache_update[class_name] = current_time
        
        return result
    
    async def _save_annotated_image_async(self, original_path: str, image: np.ndarray, face_locations: List, identified_students: List):
        """Async save annotated image - non-blocking"""
        try:
            # Create annotated image
            annotated_image = image.copy()
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Draw rectangle around face
                cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add label if student identified
                if i < len(identified_students):
                    student = identified_students[i]
                    label = f"{student['name']} ({student['confidence']:.2f})"
                    cv2.putText(annotated_image, label, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(annotated_image, "Unknown", (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Save annotated image
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            annotated_path = f"data/attendance_images/annotated_{base_name}.jpg"
            cv2.imwrite(annotated_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error saving annotated image: {e}")
    
    async def _process_with_yolov8_only(self, image_path: str, class_name: str) -> Dict:
        """
        Fallback method for cloud deployment without face_recognition
        Uses YOLOv8 detection + smart attendance marking
        """
        try:
            from app.services.cloud_face_recognition import CloudFaceRecognitionService
            
            print("🌐 Using cloud-optimized face detection")
            cloud_service = CloudFaceRecognitionService(self.student_service)
            result = await cloud_service.process_attendance_image(image_path, class_name)
            
            print(f"✅ Cloud attendance processed: {len(result['present'])} present, {len(result['absent'])} absent")
            return result
            
        except Exception as e:
            print(f"❌ Error in cloud face recognition: {e}")
            # Fallback: basic detection
            try:
                from ultralytics import YOLO
                
                model_path = "app/yolov8n.pt" if os.path.exists("app/yolov8n.pt") else "yolov8n.pt"
                model = YOLO(model_path)
                
                results = model(image_path, conf=0.3, verbose=False)
                
                num_faces = 0
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None:
                        num_faces = len(boxes)
                
                print(f"✅ Detected {num_faces} people using basic YOLO")
                
                all_students = await self.student_service.get_students(class_name)
                absent = [{'student_id': s['student_id'], 'name': s['name']} for s in all_students]
                
                return {
                    'present': [],
                    'absent': absent,
                    'total_faces_detected': num_faces,
                    'warning': f'Detected {num_faces} people but cannot identify students without face_recognition library.'
                }
            except Exception as e2:
                print(f"❌ Complete fallback failed: {e2}")
                all_students = await self.student_service.get_students(class_name)
                return {
                    'present': [],
                    'absent': [{'student_id': s['student_id'], 'name': s['name']} for s in all_students],
                    'total_faces_detected': 0,
                    'error': str(e2)
                }
    
    def clear_cache(self):
        """Clear all caches"""
        # Clear YOLOv8 cache if using it
        if self.use_insightface and self.yolov8_service:
            self.yolov8_service.clear_cache()
        
        # Clear legacy cache
        self._class_students_cache.clear()
        self._class_encodings_cache.clear()
        self._last_cache_update.clear()
        print("Face recognition cache cleared")