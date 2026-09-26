# AttendAI — Research Configuration Specification

Status: Phase 1 / research configuration freeze draft
Branch: research/phase1-insightface-architecture

This document records parameters evidenced by the research branch code and separates them from parameters that still require calibration. It is intended to prevent undocumented configuration changes during experiments.

## 1. Confirmed primary recognition stack

| Parameter | Current implementation | Research status |
|---|---|---|
| Face-processing framework | InsightFace FaceAnalysis | Frozen |
| Recognition model | buffalo_l | Frozen pending explicit model-version capture |
| Embedding | face.embedding | Frozen |
| Embedding normalization | L2 normalization | Frozen |
| Matching metric | Cosine similarity | Frozen |
| Primary attendance detector | InsightFace detector inside FaceAnalysis.get() | Frozen |
| YOLO | Model is initialized, but primary attendance detection does not call YOLO | Not part of baseline |
| Detector input size | 640 x 640 | Frozen for current implementation |
| Runtime providers | CUDA + CPU fallback when CUDA is available; CPU otherwise | Frozen as implementation behavior |

## 2. Current thresholds evidenced in code

| Parameter | Current value | Meaning | Research treatment |
|---|---:|---|---|
| Similarity threshold | Configurable; code default/fallback currently includes 0.40 | Identity-match cutoff | Calibrate experimentally |
| Detection confidence | Configurable; service fallback 0.50 | Filters detected faces | Record and freeze for each experiment |
| Minimum recognition confidence | 0.45 fallback | Legacy/minimum confidence setting | Audit whether active in canonical path |
| Verification threshold | 0.60 fallback | Borderline verification boundary | Calibrate experimentally |
| Verification top candidates | 3 | Candidates shown to teacher | Freeze after audit |
| Verification minimum quality | 0.80 fallback | Minimum quality for learning | Calibrate/validate |
| Verification auto-approve | 0.75 fallback | Auto-approval configuration | Audit whether active in attendance path |

A value appearing as a code fallback is not evidence that it is the value used in a particular experiment. The exact runtime configuration must be captured with every experiment.

## 3. Quality assessment

The current quality assessor computes four components:
- blur
- brightness
- face size
- contrast

The implementation combines them as:
Q = 0.35 B_blur + 0.25 B_brightness + 0.30 B_size + 0.10 B_contrast

Research treatment:
1. Keep the current quality computation fixed while evaluating the baseline implementation.
2. Treat the quality threshold as an experimental parameter.
3. Do not describe quality-weighted confidence as the current identity decision rule.
4. If a quality-aware decision rule is introduced, define it as a separate ablation and record its exact formula and threshold.

## 4. Enrollment representation

The current canonical service writes a generated enrollment embedding to the student's stored representation. The service also supports multiple encodings through later attendance-derived updates.

For reproducible experiments, the enrollment protocol must be explicitly fixed:
- number of enrollment images per student;
- whether enrollment images come from photographs or video frames;
- quality threshold;
- pose/occlusion restrictions;
- whether multiple enrollment embeddings are retained;
- preprocessing applied before embedding.

Until these are fixed, reported recognition results are not considered reproducible.

## 5. Adaptive encoding

Teacher-approved attendance faces can be added to a student's encoding set when the relevant continual-learning settings permit it.

For research evaluation:
- Baseline experiments: adaptation disabled.
- Adaptation experiment: enabled only after the held-out prediction has been finalized.
- Test images must never be used to update the gallery before scoring.
- The exact maximum encodings per student and consistency threshold must be recorded.

This is a leakage-control requirement, not an optional reporting detail.

## 6. Runtime environment to capture

Every benchmark should record:
- OS;
- Python version;
- PyTorch version;
- InsightFace version;
- ONNX Runtime provider/version;
- OpenCV version;
- model name and model files;
- CPU model;
- GPU model and VRAM;
- CUDA version, where applicable;
- input image resolution;
- detector input size;
- recognition threshold;
- quality threshold;
- verification threshold;
- enrollment protocol;
- dataset/split identifier.

## 7. Configuration file policy

The production/local file data/app_settings.json is not treated as research evidence by itself because it is not currently version-controlled in the research branch.

Before experiments, create a version-controlled, secret-free research configuration containing the exact values used for the experiment. Secrets, database credentials, tokens, and private paths must not be committed.

A configuration snapshot should be stored alongside experiment results so that a reported result can be reproduced from the same parameter set.

## 8. Open configuration audit

The following items must be resolved before Phase 2 experiments:
1. Identify the exact runtime data/app_settings.json used by the current application.
2. Remove stale useYOLOv8 naming from the canonical configuration path, or explicitly document it as a backward-compatible alias.
3. Remove/repair stale legacy references in FaceRecognitionService such as self.yolov8_service if that compatibility service is no longer intended to own the path.
4. Confirm the exact active similarity threshold at runtime.
5. Confirm whether detection confidence affects the canonical attendance result.
6. Confirm the exact verification interval and whether auto-approval is active.
7. Confirm enrollment encoding count per student.
8. Confirm whether adaptive encoding is enabled in the user's current local configuration.

## 9. Research rule

No numerical performance claim is considered a research result merely because it appears in README, project documentation, or a default configuration. A result becomes publishable evidence only after the dataset, split, configuration, evaluation protocol, and metric definition are recorded and the experiment is reproducibly executed.
