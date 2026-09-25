# Phase 1 — System Stabilization Audit

## Decision

The research implementation will use **InsightFace FaceAnalysis / ArcFace as the primary face-processing pipeline**.

## Current findings

- `backend/app/services/yolov8_face_recognition.py` initializes both an Ultralytics YOLO model and InsightFace.
- The attendance detection method named `detect_faces_yolov8()` actually calls `self.face_analyzer.get(image)`, i.e. InsightFace's face detector.
- The primary attendance recognition flow also uses `self.face_analyzer.get(image)` and ArcFace embeddings.
- A separate `yolov8_attendance.py` implementation exists, but it is not treated as the canonical research path.
- Identity matching uses normalized embeddings and cosine similarity with a configurable threshold.
- Face quality assessment is implemented, but the quality-weighted confidence value is not currently the identity decision criterion.
- Teacher verification and verified encoding updates exist as optional downstream mechanisms.

## Research architecture rule

Do not write the paper as a YOLOv8 + ArcFace attendance pipeline at this stage. The defensible description is InsightFace FaceAnalysis / ArcFace-based face detection and recognition, with quality assessment, human verification, and optional verified-encoding adaptation.

## Known stabilization work

1. Keep one canonical attendance recognition path.
2. Rename or refactor misleading YOLO-named methods/classes so names reflect actual behavior.
3. Keep the separate YOLO attendance implementation isolated from research experiments unless deliberately re-integrated.
4. Fix endpoint/service mismatches before benchmarking.
5. Update tests to match current quality and landmark APIs.
6. Add a version-controlled research configuration without secrets.
7. Add architecture and research-mode documentation.

## Research safety rule

Adaptive encoding updates must never change the representation set before a held-out test prediction is scored. Evaluation runs must record whether adaptation is disabled, frozen, or applied only between sessions.


## Phase 1 migration update

- A canonical service boundary was added at `backend/app/services/insightface_face_recognition.py`.
- Active enrollment, rebuild, settings, student, verification, optimized-attendance, and continual-learning integrations were migrated to the canonical InsightFace boundary.
- The optimized attendance endpoint now calls the implemented `process_attendance_image()` path rather than the obsolete `process_attendance()` call.
- The obsolete `optimized_processor` references in the optimized endpoint were removed from the active paths.
- The historical `yolov8_face_recognition.py` implementation remains temporarily as a compatibility implementation; it has not yet been deleted or renamed because that would be a larger behavior-changing refactor.
- This migration therefore establishes the research-facing service boundary without claiming that all historical implementation terminology has already been removed.
