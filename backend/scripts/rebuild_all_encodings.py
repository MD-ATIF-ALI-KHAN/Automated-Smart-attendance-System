"""
Rebuild YOLOv8 Face Encodings for All Students (using InsightFace/ArcFace)
Run this to enroll all students with high-quality face embeddings
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend root to path and set working directory
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
os.chdir(str(BACKEND_ROOT))

import cv2
import numpy as np
from app.services.insightface_face_recognition import InsightFaceFaceRecognitionService
from app.services.student_management import StudentManagementService


async def main():
    print("=" * 60)
    print("🔄 Rebuilding InsightFace (ArcFace) Encodings for All Students")
    print("=" * 60)
    
    # Initialize services
    student_service = StudentManagementService()
    face_service = InsightFaceFaceRecognitionService()
    await face_service.load_encodings()

    # Increase detector input size to help detect smaller faces in large frames
    try:
        ctx_id = 0 if face_service.device == 'cuda' else -1
        face_service.face_analyzer.prepare(ctx_id=ctx_id, det_size=(1280, 1280))
    except Exception:
        pass
    
    # Get student IDs from image directories
    image_dir = "data/student_images"
    student_folders = [d for d in os.listdir(image_dir) 
                      if os.path.isdir(os.path.join(image_dir, d)) and d.isdigit()]
    
    print(f"\n📋 Found {len(student_folders)} students with images: {', '.join(student_folders)}")
    print("-" * 60)
    
    success_count = 0
    failed_count = 0
    
    # Enroll each student using MULTIPLE images
    for student_id in student_folders:
        student_path = os.path.join(image_dir, student_id)
        images = [f for f in os.listdir(student_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"❌ Student {student_id}: No images found")
            failed_count += 1
            continue
        
        # Generate encodings from MULTIPLE images and average them
        encodings_list = []
        images_to_use = images[:min(5, len(images))]  # Use up to 5 images
        
        for img_file in images_to_use:
            image_path = os.path.join(student_path, img_file)
            try:
                # Generate encoding WITHOUT saving (we'll save the average later)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # Try enhanced image first, then fallback to original
                import cv2
                enhanced = face_service._enhance_image(image)
                faces = face_service.face_analyzer.get(enhanced)

                if len(faces) == 0:
                    faces = face_service.face_analyzer.get(image)

                if len(faces) == 0:
                    # Upscale small images and retry
                    upscaled = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    faces = face_service.face_analyzer.get(upscaled)

                if len(faces) > 0:
                    # Get best face embedding
                    faces.sort(key=lambda x: x.det_score, reverse=True)
                    embedding = faces[0].embedding
                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                        encodings_list.append(embedding)
            except Exception as e:
                continue
        
        if len(encodings_list) > 0:
            # Average all encodings for robustness
            avg_encoding = np.mean(encodings_list, axis=0)
            # Re-normalize the average
            norm = np.linalg.norm(avg_encoding)
            if norm > 0:
                avg_encoding = avg_encoding / norm
            
            # Store the averaged encoding
            face_service.known_face_encodings[student_id] = avg_encoding
            
            # Get student info
            student = await face_service.student_service.get_student_by_id(student_id)
            if student:
                face_service.known_face_names[student_id] = student['name']
            
            print(f"✅ Student {student_id}: Enrolled (averaged {len(encodings_list)} images from {len(images)} available)")
            success_count += 1
        else:
            print(f"❌ Student {student_id}: No faces detected in any image")
            failed_count += 1
    
    # Save all encodings
    await face_service.save_encodings()
    
    print("\n" + "=" * 60)
    print("📊 ENROLLMENT RESULTS")
    print("=" * 60)
    print(f"✅ Successfully enrolled: {success_count} students")
    print(f"❌ Failed enrollments: {failed_count} students")
    
    print("\n" + "=" * 60)
    print(f"✅ Encodings saved to: {face_service.encodings_file}")
    print("=" * 60)
    
    # Verify encoding file
    import pickle
    import numpy as np
    if os.path.exists(face_service.encodings_file):
        with open(face_service.encodings_file, 'rb') as f:
            data = pickle.load(f)
        encodings = data.get('encodings', {})
        print(f"\n✅ Verified: {len(encodings)} students in encoding file")
        print(f"   Students: {', '.join(encodings.keys())}")
        # Check normalization
        for sid, enc in encodings.items():
            norm = np.linalg.norm(enc)
            print(f"   Student {sid} norm: {norm:.6f} {'✓' if abs(norm - 1.0) < 0.01 else '✗ NOT NORMALIZED'}")
    else:
        print("\n⚠️  Warning: Encoding file not found!")


if __name__ == "__main__":
    asyncio.run(main())
