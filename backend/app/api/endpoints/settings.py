"""
Settings API Endpoints
Handles loading and saving face recognition settings
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio

import cv2
import numpy as np

from app.services.student_management import StudentManagementService

router = APIRouter()

_rebuild_lock = asyncio.Lock()
_rebuild_status: Dict[str, Any] = {
    "status": "idle",
    "current": 0,
    "total": 0,
    "message": "",
    "last_error": None,
    "started_at": None,
    "finished_at": None,
    "student_id": None,
    "current_student": None,
    "success_count": 0,
    "failed_count": 0,
}

class SettingsModel(BaseModel):
    fpsExtraction: int = 2
    similarityThreshold: float = 0.25
    detectionConfidence: float = 0.25
    useGPU: bool = False


class RebuildEncodingsRequest(BaseModel):
    student_id: Optional[str] = None

SETTINGS_FILE = "data/app_settings.json"


def _create_insightface_face_service():
    """Create the canonical InsightFace service on-demand.

    This endpoint module is imported at app startup. ML deps like torch/ultralytics
    are optional in some deployments, so we avoid importing them globally.
    """
    try:
        from app.services.insightface_face_recognition import InsightFaceFaceRecognitionService
    except Exception as exc:  # ImportError + runtime import errors
        raise HTTPException(
            status_code=503,
            detail=(
                "InsightFace/ArcFace recognition is not available in this environment. "
                "Install the ML dependencies (e.g. torch, insightface) "
                "and restart the server."
            ),
        ) from exc

    return InsightFaceFaceRecognitionService()


def _reset_rebuild_status():
    _rebuild_status.update({
        "status": "idle",
        "current": 0,
        "total": 0,
        "message": "",
        "last_error": None,
        "started_at": None,
        "finished_at": None,
        "student_id": None,
        "current_student": None,
        "success_count": 0,
        "failed_count": 0,
    })


def _set_rebuild_status(**updates: Any):
    _rebuild_status.update(updates)


async def _rebuild_encodings(student_id: Optional[str]):
    student_service = StudentManagementService()
    face_service = _create_insightface_face_service()

    if face_service.face_analyzer is None:
        raise RuntimeError("InsightFace not initialized")

    await face_service.load_encodings()

    if not student_id:
        face_service.known_face_encodings = {}
        face_service.known_face_names = {}

    image_dir = "data/student_images"
    if student_id:
        student_folders = [student_id]
    else:
        student_folders = [
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ]

    _set_rebuild_status(
        status="running",
        current=0,
        total=len(student_folders),
        message="Rebuilding encodings",
        last_error=None,
        started_at=datetime.now().isoformat(),
        finished_at=None,
        student_id=student_id,
        current_student=None,
        success_count=0,
        failed_count=0,
    )

    success_count = 0
    failed_count = 0
    processed = 0

    for current_student_id in student_folders:
        _set_rebuild_status(current_student=current_student_id)
        student_path = os.path.join(image_dir, current_student_id)
        if not os.path.isdir(student_path):
            failed_count += 1
            processed += 1
            _set_rebuild_status(current=processed, failed_count=failed_count)
            continue

        images = [
            f for f in os.listdir(student_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not images:
            failed_count += 1
            processed += 1
            _set_rebuild_status(current=processed, failed_count=failed_count)
            continue

        valid_encodings = []

        for img_file in images[:10]:
            image_path = os.path.join(student_path, img_file)
            image = cv2.imread(image_path)
            if image is None:
                continue

            enhanced = face_service._enhance_image(image)
            faces = face_service.face_analyzer.get(enhanced)
            if not faces:
                continue

            if len(faces) > 1:
                faces.sort(key=lambda x: x.det_score, reverse=True)

            embedding = faces[0].embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                valid_encodings.append(embedding / norm)

        if valid_encodings:
            avg_encoding = np.mean(valid_encodings, axis=0)
            norm = np.linalg.norm(avg_encoding)
            if norm > 0:
                avg_encoding = avg_encoding / norm

            face_service.known_face_encodings[current_student_id] = avg_encoding

            student = await student_service.get_student_by_id(current_student_id)
            if student:
                face_service.known_face_names[current_student_id] = student["name"]
            else:
                face_service.known_face_names[current_student_id] = f"Student {current_student_id}"

            success_count += 1
        else:
            failed_count += 1

        processed += 1
        _set_rebuild_status(current=processed, success_count=success_count, failed_count=failed_count)

    await face_service.save_encodings()

    _set_rebuild_status(
        status="done",
        message="Rebuild complete",
        finished_at=datetime.now().isoformat(),
        success_count=success_count,
        failed_count=failed_count,
        current_student=None,
    )

@router.get("/settings")
async def get_settings():
    """Get current face recognition settings"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                return {
                    "fpsExtraction": settings.get("fpsExtraction", 2),
                    "similarityThreshold": settings.get("insightfaceRecognition", {}).get("similarityThreshold", 0.25),
                    "detectionConfidence": settings.get("yolov8Detection", {}).get("confidence", 0.25),
                    "useGPU": settings.get("useGPU", False)
                }
        else:
            # Return defaults
            return {
                "fpsExtraction": 2,
                "similarityThreshold": 0.25,
                "detectionConfidence": 0.25,
                "useGPU": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings")
async def save_settings(settings: SettingsModel):
    """Save face recognition settings"""
    try:
        # Load existing settings or create new
        existing_settings = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                existing_settings = json.load(f)
        
        # Update settings
        existing_settings["fpsExtraction"] = settings.fpsExtraction
        existing_settings["useGPU"] = settings.useGPU
        
        # Update YOLO detection settings
        if "yolov8Detection" not in existing_settings:
            existing_settings["yolov8Detection"] = {}
        existing_settings["yolov8Detection"]["confidence"] = settings.detectionConfidence
        existing_settings["yolov8Detection"]["iouThreshold"] = 0.45
        
        # Update InsightFace recognition settings
        if "insightfaceRecognition" not in existing_settings:
            existing_settings["insightfaceRecognition"] = {}
        existing_settings["insightfaceRecognition"]["model"] = "buffalo_l"
        existing_settings["insightfaceRecognition"]["distanceMetric"] = "cosine"
        existing_settings["insightfaceRecognition"]["similarityThreshold"] = settings.similarityThreshold
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        
        # Save to file
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(existing_settings, f, indent=2)
        
        return {"message": "Settings saved successfully", "settings": settings}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/settings/rebuild-encodings")
async def rebuild_encodings(request: RebuildEncodingsRequest):
    """Rebuild face encodings for all students or a specific student"""
    async with _rebuild_lock:
        if _rebuild_status.get("status") == "running":
            return {"status": "running", "message": "Rebuild already in progress", "progress": _rebuild_status}

        _reset_rebuild_status()

        async def _runner():
            try:
                await _rebuild_encodings(request.student_id)
            except Exception as exc:
                _set_rebuild_status(
                    status="error",
                    message="Rebuild failed",
                    last_error=str(exc),
                    finished_at=datetime.now().isoformat(),
                )

        asyncio.create_task(_runner())

    return {"status": "started", "progress": _rebuild_status}


@router.get("/settings/rebuild-encodings/status")
async def rebuild_encodings_status():
    """Get rebuild encodings progress"""
    return _rebuild_status
