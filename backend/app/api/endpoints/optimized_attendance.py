"""
Optimized Attendance API Endpoints
High-performance attendance marking endpoints using the canonical InsightFace/ArcFace recognition pipeline
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional, Dict, List
import os
import asyncio
from datetime import datetime
import time

try:
    # Optional dependency: requires torch + ultralytics + insightface, etc.
    from app.services.insightface_face_recognition import insightface_face_service
    _INSIGHTFACE_SERVICE_AVAILABLE = True
except ImportError:
    insightface_face_service = None  # type: ignore
    _INSIGHTFACE_SERVICE_AVAILABLE = False
from app.services.attendance import AttendanceService
from app.services.student_management import StudentManagementService
from app.services.face_recognition import FaceRecognitionService

router = APIRouter()

# Initialize canonical InsightFace/ArcFace service (singleton)
_insightface_service = None

def get_insightface_service():
    global _insightface_service
    if not _INSIGHTFACE_SERVICE_AVAILABLE or insightface_face_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "InsightFace optimized attendance is not available in this environment. "
                "Install the ML dependencies (e.g. torch, insightface) and restart the server."
            ),
        )
    if _insightface_service is None:
        _insightface_service = insightface_face_service
        _insightface_service.similarity_threshold = 0.50
        print("✅ Canonical InsightFace (ArcFace) service initialized with 0.50 threshold")
    return _insightface_service

@router.post("/mark-attendance-optimized")
async def mark_attendance_optimized(
    background_tasks: BackgroundTasks,
    class_name: str = Form(...),
    image: UploadFile = File(...)
) -> Dict:
    """
    Optimized attendance marking endpoint with performance improvements
    """
    try:
        start_time = time.time()
        
        # Validate file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"attendance_{class_name}_{timestamp}.jpg"
        image_path = f"data/attendance_images/{image_filename}"
        
        # Ensure directory exists
        os.makedirs("data/attendance_images", exist_ok=True)
        
        # Save image
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        print(f"Processing attendance for class {class_name}...")
        
        # Use canonical InsightFace (ArcFace) pipeline for identification
        face_service = get_insightface_service()
        result = await face_service.process_attendance_image(image_path, class_name)
        
        # Save attendance record
        attendance_service = AttendanceService()
        attendance_id = await attendance_service.save_attendance(
            class_name=class_name,
            image_path=image_path,
            present_students=result['present'],
            absent_students=result['absent'],
            total_faces_detected=result['total_faces_detected']
        )
        
        # Background task: Clean up old images
        background_tasks.add_task(cleanup_old_images)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "attendance_id": attendance_id,
            "class_name": class_name,
            "present_count": len(result['present']),
            "absent_count": len(result['absent']),
            "total_faces_detected": result['total_faces_detected'],
            "present_students": result['present'],
            "absent_students": result['absent'],
            "pending_verifications": result.get('pending_verifications', []),
            "has_pending_verifications": result.get('has_pending_verifications', False),
            "processing_time": round(processing_time, 3),
            "image_path": image_path
        }
        
    except Exception as e:
        print(f"Error in optimized attendance marking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-metrics")
async def get_performance_metrics() -> Dict:
    """Get performance metrics for the YOLOv8 system"""
    try:
        face_service = get_insightface_service()
        
        # Get InsightFace stats
        stats = {
            "total_students_enrolled": len(yolov8_service.face_descriptors),
            "model_status": "YOLOv8n (CPU mode)",
            "encoding_file": "yolov8_face_encodings.pkl",
            "system_status": "YOLOv8 face matching active"
        }
        
        # Get face recognition stats if available
        try:
            face_service = FaceRecognitionService()
            face_stats = await face_service.get_recognition_stats()
            stats["legacy_face_service"] = face_stats
        except:
            pass
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-caches")
async def clear_all_caches() -> Dict:
    """Clear all system caches"""
    try:
        # Clear YOLOv8 cache if needed
        yolov8_service = get_yolov8_service()
        yolov8_service.face_descriptors.clear()
        yolov8_service.student_id_map.clear()
        
        # Reload encodings
        yolov8_service._load_encodings()
        
        return {
            "success": True,
            "message": "YOLOv8 encodings reloaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-attendance-optimized")
async def batch_attendance_optimized(
    background_tasks: BackgroundTasks,
    class_name: str = Form(...),
    images: List[UploadFile] = File(...)
) -> Dict:
    """
    Process multiple attendance images in batch - OPTIMIZED
    """
    try:
        start_time = time.time()
        results = []
        
        # Process images in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent processes
        
        async def process_single_image(image: UploadFile, index: int):
            async with semaphore:
                try:
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"batch_{class_name}_{index}_{timestamp}.jpg"
                    image_path = f"data/attendance_images/{image_filename}"
                    
                    with open(image_path, "wb") as buffer:
                        content = await image.read()
                        buffer.write(content)
                    
                    # Process attendance
                    result = await face_service.process_attendance_image(image_path, class_name)
                    
                    # Save attendance record
                    attendance_service = AttendanceService()
                    attendance_id = await attendance_service.save_attendance(
                        class_name=class_name,
                        image_path=image_path,
                        present_students=result['present'],
                        absent_students=result['absent'],
                        total_faces_detected=result['total_faces_detected']
                    )
                    
                    return {
                        "index": index,
                        "attendance_id": attendance_id,
                        "present_count": len(result['present']),
                        "absent_count": len(result['absent']),
                        "processing_time": result.get('processing_time', 0),
                        "success": True
                    }
                    
                except Exception as e:
                    return {
                        "index": index,
                        "error": str(e),
                        "success": False
                    }
        
        # Process all images concurrently
        tasks = [process_single_image(image, i) for i, image in enumerate(images)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Background cleanup
        background_tasks.add_task(cleanup_old_images)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        
        return {
            "success": True,
            "total_images": len(images),
            "successful": successful,
            "failed": len(images) - successful,
            "total_processing_time": round(total_time, 3),
            "results": results
        }
        
    except Exception as e:
        print(f"Error in batch attendance processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_old_images():
    """Background task to clean up old attendance images"""
    try:
        attendance_images_dir = "data/attendance_images"
        if not os.path.exists(attendance_images_dir):
            return
        
        # Remove images older than 7 days
        current_time = time.time()
        cutoff_time = current_time - (7 * 24 * 60 * 60)  # 7 days
        
        for filename in os.listdir(attendance_images_dir):
            file_path = os.path.join(attendance_images_dir, filename)
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                if file_time < cutoff_time:
                    os.remove(file_path)
                    print(f"Cleaned up old image: {filename}")
        
    except Exception as e:
        print(f"Error in cleanup task: {e}")

@router.get("/system-status")
async def get_system_status() -> Dict:
    """Get comprehensive system status"""
    try:
        # Get performance metrics
        metrics = {"service": "InsightFace/ArcFace", "model": face_service.face_recognition_model}
        
        # Get face recognition stats
        face_service = FaceRecognitionService()
        face_stats = await face_service.get_recognition_stats()
        
        # Get attendance statistics
        attendance_service = AttendanceService()
        attendance_stats = await attendance_service.get_statistics()
        
        return {
            "system_status": "canonical-insightface",
            "performance_metrics": metrics,
            "face_recognition": face_stats,
            "attendance_stats": attendance_stats,
            "optimizations_active": [
                "Smart image enhancement",
                "Caching system",
                "Batch database operations",
                "Parallel processing",
                "Background cleanup"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detection-settings")
async def get_detection_settings() -> Dict:
    """Get current face detection settings"""
    try:
        import json
        settings_file = "data/app_settings.json"
        
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                return {
                    "success": True,
                    "settings": settings.get('faceDetection', {}),
                    "available_modes": {
                        "fast": {
                            "description": "Fastest processing, good for close-up faces",
                            "upsampleTimes": 0,
                            "detectionScales": [1.0],
                            "useCNN": False,
                            "numJitters": 1
                        },
                        "balanced": {
                            "description": "Good balance of speed and accuracy (recommended)",
                            "upsampleTimes": 0,
                            "detectionScales": [1.0, 1.5],
                            "useCNN": False,
                            "numJitters": 3
                        },
                        "accurate": {
                            "description": "Best accuracy for distant faces, slower processing",
                            "upsampleTimes": 1,
                            "detectionScales": [1.0, 1.5, 2.0],
                            "useCNN": True,
                            "numJitters": 5
                        }
                    }
                }
        else:
            return {
                "success": False,
                "message": "Settings file not found"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detection-settings")
async def update_detection_settings(settings: Dict) -> Dict:
    """Update face detection settings"""
    try:
        import json
        settings_file = "data/app_settings.json"
        
        # Load current settings
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                app_settings = json.load(f)
        else:
            app_settings = {}
        
        # Update face detection settings
        face_detection = settings.get('faceDetection', {})
        
        # Validate mode if provided
        valid_modes = ['fast', 'balanced', 'accurate']
        mode = face_detection.get('mode', 'balanced')
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
        
        # Validate upsampleTimes
        upsample_times = face_detection.get('upsampleTimes', 0)
        if not isinstance(upsample_times, int) or upsample_times < 0 or upsample_times > 3:
            raise HTTPException(status_code=400, detail="upsampleTimes must be an integer between 0 and 3")
        
        # Validate detectionScales
        detection_scales = face_detection.get('detectionScales', [1.0, 1.5])
        if not isinstance(detection_scales, list) or not all(isinstance(s, (int, float)) and s >= 1.0 for s in detection_scales):
            raise HTTPException(status_code=400, detail="detectionScales must be a list of numbers >= 1.0")
        
        # Validate numJitters
        num_jitters = face_detection.get('numJitters', 3)
        if not isinstance(num_jitters, int) or num_jitters < 1 or num_jitters > 10:
            raise HTTPException(status_code=400, detail="numJitters must be an integer between 1 and 10")
        
        # Update settings
        app_settings['faceDetection'] = face_detection
        
        # Save settings
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        with open(settings_file, 'w') as f:
            json.dump(app_settings, f, indent=2)
        
        # Clear caches to force reload with new settings
        optimized_processor.clear_all_caches()
        
        # Reload settings in face service
        face_service = FaceRecognitionService()
        await face_service.load_encodings()
        
        return {
            "success": True,
            "message": "Detection settings updated successfully. Changes will take effect on next attendance marking.",
            "settings": face_detection,
            "note": "Server restart recommended for full effect"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recognition-settings")
async def get_recognition_settings() -> Dict:
    """Get current face recognition accuracy settings"""
    try:
        import json
        settings_file = "data/app_settings.json"
        
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                return {
                    "success": True,
                    "settings": settings.get('faceRecognition', {}),
                    "recommendations": {
                        "strict": {
                            "description": "Best accuracy, prevents false matches (recommended)",
                            "tolerance": 0.45,
                            "minConfidence": 0.50,
                            "strictMode": True
                        },
                        "balanced": {
                            "description": "Good balance between accuracy and detection rate",
                            "tolerance": 0.50,
                            "minConfidence": 0.45,
                            "strictMode": True
                        },
                        "lenient": {
                            "description": "More detections, but higher chance of false matches",
                            "tolerance": 0.55,
                            "minConfidence": 0.40,
                            "strictMode": False
                        }
                    }
                }
        else:
            return {
                "success": False,
                "message": "Settings file not found"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognition-settings")
async def update_recognition_settings(settings: Dict) -> Dict:
    """Update face recognition accuracy settings"""
    try:
        import json
        settings_file = "data/app_settings.json"
        
        # Load current settings
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                app_settings = json.load(f)
        else:
            app_settings = {}
        
        # Update face recognition settings
        face_recognition = settings.get('faceRecognition', {})
        
        # Validate tolerance
        tolerance = face_recognition.get('tolerance', 0.45)
        if not isinstance(tolerance, (int, float)) or tolerance < 0.3 or tolerance > 0.7:
            raise HTTPException(status_code=400, detail="tolerance must be a number between 0.3 and 0.7 (lower = stricter)")
        
        # Validate minConfidence
        min_confidence = face_recognition.get('minConfidence', 0.50)
        if not isinstance(min_confidence, (int, float)) or min_confidence < 0.3 or min_confidence > 0.9:
            raise HTTPException(status_code=400, detail="minConfidence must be a number between 0.3 and 0.9")
        
        # Validate strictMode
        strict_mode = face_recognition.get('strictMode', True)
        if not isinstance(strict_mode, bool):
            raise HTTPException(status_code=400, detail="strictMode must be a boolean")
        
        # Update settings
        app_settings['faceRecognition'] = face_recognition
        
        # Save settings
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        with open(settings_file, 'w') as f:
            json.dump(app_settings, f, indent=2)
        
        # Clear caches to force reload with new settings
        optimized_processor.clear_all_caches()
        
        # Reload settings in face service
        face_service = FaceRecognitionService()
        await face_service.load_encodings()
        
        return {
            "success": True,
            "message": "Recognition settings updated successfully. Changes will take effect on next attendance marking.",
            "settings": face_recognition,
            "note": "Server restart recommended for full effect",
            "impact": {
                "tolerance": f"{'Stricter' if tolerance < 0.5 else 'More lenient'} matching ({tolerance})",
                "minConfidence": f"Minimum {min_confidence*100:.0f}% confidence required",
                "strictMode": "Both conditions must be met" if strict_mode else "Either condition can be met"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

