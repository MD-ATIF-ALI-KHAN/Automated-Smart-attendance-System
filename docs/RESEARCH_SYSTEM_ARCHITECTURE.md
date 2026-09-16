# AttendAI — Final Research System Architecture

**Architecture status:** Phase 1 — architecture freeze (research branch)

## 1. Frozen primary pipeline

The research version of AttendAI uses **InsightFace FaceAnalysis with the ArcFace recognition model as the primary face-processing pipeline**. The Ultralytics YOLO service remains in the repository as a separate/legacy component and is **not** part of the primary attendance recognition path unless explicitly re-integrated and evaluated.

```text
                    CLASSROOM IMAGE / CAMERA FRAME
                                  |
                                  v
                     +--------------------------+
                     | Image acquisition        |
                     | upload / scheduled cam   |
                     +------------+-------------+
                                  |
                                  v
                     +--------------------------+
                     | Input / image validation |
                     +------------+-------------+
                                  |
                                  v
               +---------------------------------------+
               | InsightFace FaceAnalysis              |
               | face detection + face analysis        |
               +-------------------+-------------------+
                                   |
                       detected face(s)
                                   |
                +------------------+------------------+
                |                                     |
                v                                     v
     +-----------------------+             +-----------------------+
     | Face quality          |             | ArcFace embedding     |
     | assessment            |             | extraction            |
     | blur/brightness/size/ |             | normalized embedding  |
     | contrast              |             +-----------+-----------+
     +-----------+-----------+                         |
                 |                                     |
                 +------------------+------------------+
                                    |
                                    v
                       +--------------------------+
                       | Identity matching        |
                       | cosine similarity        |
                       | against enrolled        |
                       | student encodings       |
                       +------------+-------------+
                                    |
                       +------------+-------------+
                       |                          |
                 confident match             ambiguous/borderline
                       |                          |
                       v                          v
              +----------------+       +-----------------------+
              | Attendance      |       | Teacher verification  |
              | candidate       |       | approve / reject /    |
              +--------+-------+       | unknown                |
                       |               +-----------+-----------+
                       |                           |
                       +-------------+-------------+
                                     |
                                     v
                         +-------------------------+
                         | Attendance persistence  |
                         | session / student /     |
                         | timestamp / confidence  |
                         +------------+------------+
                                      |
                                      v
                         +-------------------------+
                         | Reports / analytics     |
                         +-------------------------+

Optional controlled research path:
Teacher-approved, high-quality attendance face
                    |
                    v
          verified encoding update
                    |
                    v
       enrolled encoding set (future sessions)

NOTE: Adaptive encoding updates must be disabled or frozen
when evaluating a held-out test set, then applied only after
that test prediction has been finalized, to prevent leakage.
```

## 2. Component-to-code mapping

| Layer | Current implementation evidence | Research role |
|---|---|---|
| Recognition service | `backend/app/services/yolov8_face_recognition.py` | Canonical service containing InsightFace/ArcFace attendance logic |
| Face analysis | `FaceAnalysis(name='buffalo_l')` | Primary detector/face analysis and ArcFace embedding source |
| Embedding | `face.embedding`, L2 normalization | Identity representation |
| Matching | cosine similarity + configured threshold | Identity decision baseline |
| Quality | `app/utils/face_quality.py` | Input/attendance quality assessment |
| Verification | `app/utils/verification_manager.py` + verification flow | Human-in-the-loop handling of uncertain cases |
| Attendance | attendance service/database layer | Persistent attendance record |
| Automation | scheduled attendance service | Repeated camera acquisition and processing |
| Adaptation | attendance-derived verified encodings | Optional continual/incremental representation update |

## 3. Important architecture decisions

### 3.1 Primary detector/recognizer

For the research paper, describe the active pipeline as **InsightFace FaceAnalysis / ArcFace-based face detection and recognition**. The current service initializes an Ultralytics YOLO model, but its attendance detection method calls `FaceAnalysis.get(image)`; therefore the repository should not claim that YOLOv8 is the active detector for attendance until a dedicated YOLO→ArcFace path is deliberately wired and benchmarked.

### 3.2 Identity decision

The current identity decision is based on the **best cosine similarity against stored student embeddings and a similarity threshold**. The quality-weighted confidence value is supplementary at this stage; it must not be described as the identity decision rule until the implementation is changed and experimentally validated.

### 3.3 Quality assessment

Quality assessment measures blur, brightness, face size, and contrast. It is used operationally for image/enrollment validation and automated capture/retry. Research evaluation will determine whether adding quality-aware decision logic improves recognition reliability.

### 3.4 Human verification

Borderline matches can enter a teacher-verification workflow. The teacher can approve, reject, or mark a case unknown; approved cases can optionally contribute a verified encoding.

### 3.5 Adaptive encoding

Verified attendance encodings may be added to a student's representation set. This is treated as an **optional experimental condition**, not an always-on component of the baseline.

## 4. Research baselines and ablations

The architecture supports the following controlled comparison:

- **B0 — Baseline:** InsightFace/ArcFace embedding + cosine similarity + fixed threshold.
- **B1 — Quality assessment:** B0 + quality assessment/gating.
- **B2 — Quality-aware decision:** B1 + experimentally defined quality-aware confidence/decision rule.
- **B3 — Human verification:** B2 + teacher verification for uncertain cases.
- **B4 — Verified encoding adaptation:** B3 + controlled addition of teacher-approved, high-quality encodings.

Each condition must use the same held-out evaluation set and the same enrollment identities. Adaptive updates must never be allowed to modify the representation using images from the held-out test set before its prediction is scored.

## 5. What is explicitly not frozen yet

The following research parameters require empirical calibration before the paper reports results:

- recognition threshold;
- verification threshold and ambiguity interval;
- minimum acceptable image/face quality;
- enrollment protocol and number of enrollment samples per student;
- dataset size and train/enrollment/test split;
- classroom condition categories;
- latency/FPS measurement protocol;
- final statistical analysis.

These are experimental parameters, not assumptions to be presented as established performance claims.

## 6. Phase 1 acceptance criteria

Phase 1 architecture is considered frozen when:

1. one canonical attendance recognition path is documented;
2. InsightFace/ArcFace is the primary recognition pipeline;
3. YOLO is not described as active attendance detection unless explicitly wired;
4. baseline and optional research components are separable;
5. test-time adaptive learning is leakage-controlled;
6. future code changes are evaluated against this architecture before being included in research experiments.
