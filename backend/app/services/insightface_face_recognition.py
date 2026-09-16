"""
Canonical InsightFace face-recognition service boundary for AttendAI.

Research architecture note:
    InsightFace FaceAnalysis (buffalo_l / ArcFace embeddings) is the canonical
    face-processing pipeline for the research system. The implementation is
    currently backed by the legacy service module for compatibility; callers
    should import this module instead of depending on the historical
    ``yolov8_face_recognition`` module directly.

This compatibility layer is intentionally small. The underlying implementation
will be cleaned up in the next architecture-refactoring step without changing
the public behavior of the active recognition pipeline.
"""

from app.services.yolov8_face_recognition import (
    YOLOv8FaceRecognitionService,
    yolov8_face_service,
)


class InsightFaceFaceRecognitionService(YOLOv8FaceRecognitionService):
    """Canonical service interface for the InsightFace/ArcFace pipeline."""

    pass


# Canonical singleton used by integrations that need the active service.
insightface_face_service = InsightFaceFaceRecognitionService()

__all__ = [
    "InsightFaceFaceRecognitionService",
    "insightface_face_service",
]
