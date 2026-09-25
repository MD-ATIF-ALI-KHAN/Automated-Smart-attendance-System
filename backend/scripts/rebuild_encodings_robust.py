"""
Robust Face Encoding Rebuild Script
Generates high-quality averaged face encodings using InsightFace
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import cv2
import numpy as np

# Add backend root to path and set working directory
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
os.chdir(str(BACKEND_ROOT))

from app.services.insightface_face_recognition import InsightFaceFaceRecognitionService
from app.services.student_management import StudentManagementService


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild face encodings for all students or a specific student."
    )
    parser.add_argument(
        "--student-id",
        dest="student_id",
        help="Rebuild encodings for a single student id (folder name in data/student_images).",
    )
    return parser.parse_args()


async def main():
    print("=" * 70)
    print("🔄 Rebuilding Face Encodings with Multi-Image Averaging")
    print("=" * 70)
    
    # Initialize services
    student_service = StudentManagementService()
    face_service = InsightFaceFaceRecognitionService()
    
    # Clear existing encodings
    face_service.known_face_encodings = {}
    face_service.known_face_names = {}
    
    args = _parse_args()

    # Get student IDs from image directories
    image_dir = "data/student_images"
    if args.student_id:
        student_folders = [args.student_id]
    else:
        student_folders = [
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ]
    
    print(f"\n📋 Found {len(student_folders)} student folders: {', '.join(student_folders)}")
    print("-" * 70)
    
    success_count = 0
    failed_count = 0
    
    # Process each student
    for student_id in student_folders:
        student_path = os.path.join(image_dir, student_id)
        if not os.path.isdir(student_path):
            print(f"❌ Student {student_id}: Folder not found at {student_path}")
            failed_count += 1
            continue
        images = [f for f in os.listdir(student_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"❌ Student {student_id}: No images found")
            failed_count += 1
            continue
        
        print(f"\n📸 Processing Student {student_id} ({len(images)} images)...")
        
        # Collect encodings from multiple images
        valid_encodings = []
        
        for i, img_file in enumerate(images[:10]):  # Use up to 10 images
            image_path = os.path.join(student_path, img_file)
            
            try:
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"   ⚠️  Could not load {img_file}")
                    continue
                
                # Enhance image
                image = face_service._enhance_image(image)
                
                # Detect faces
                faces = face_service.face_analyzer.get(image)
                
                if len(faces) == 0:
                    print(f"   ⚠️  No face in {img_file}")
                    continue
                
                # Get best face
                if len(faces) > 1:
                    faces.sort(key=lambda x: x.det_score, reverse=True)
                
                face = faces[0]
                confidence = float(face.det_score)
                
                # Get and normalize embedding
                embedding = face.embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    valid_encodings.append(embedding)
                    print(f"   ✓ {img_file}: confidence={confidence:.3f}, norm={norm:.3f}")
                
            except Exception as e:
                print(f"   ✗ Error with {img_file}: {e}")
                continue
        
        # Average encodings if we have any
        if len(valid_encodings) > 0:
            # Average all encodings
            avg_encoding = np.mean(valid_encodings, axis=0)
            
            # Re-normalize the averaged encoding
            norm = np.linalg.norm(avg_encoding)
            if norm > 0:
                avg_encoding = avg_encoding / norm
            
            # Store the averaged encoding
            face_service.known_face_encodings[student_id] = avg_encoding
            
            # Get student info from database
            try:
                student = await student_service.get_student_by_id(student_id)
                if student:
                    face_service.known_face_names[student_id] = student['name']
                    print(f"   ✅ Averaged {len(valid_encodings)} encodings for {student['name']} (ID: {student_id})")
                else:
                    face_service.known_face_names[student_id] = f"Student {student_id}"
                    print(f"   ✅ Averaged {len(valid_encodings)} encodings for Student {student_id}")
            except:
                face_service.known_face_names[student_id] = f"Student {student_id}"
                print(f"   ✅ Averaged {len(valid_encodings)} encodings for Student {student_id}")
            
            success_count += 1
        else:
            print(f"   ❌ Student {student_id}: No valid faces detected")
            failed_count += 1
    
    # Save all encodings
    await face_service.save_encodings()
    
    print("\n" + "=" * 70)
    print("📊 ENROLLMENT RESULTS")
    print("=" * 70)
    print(f"✅ Successfully enrolled: {success_count} students")
    print(f"❌ Failed enrollments: {failed_count} students")
    
    print("\n" + "=" * 70)
    print(f"✅ Encodings saved to: {face_service.encodings_file}")
    print("=" * 70)
    
    # Verify saved encodings
    import pickle
    if os.path.exists(face_service.encodings_file):
        with open(face_service.encodings_file, 'rb') as f:
            data = pickle.load(f)
        encodings = data.get('encodings', {})
        print(f"\n✅ Verified: {len(encodings)} students in encoding file")
        print(f"   Students: {', '.join(encodings.keys())}")
        
        # Check normalization
        for sid, enc in encodings.items():
            norm = np.linalg.norm(enc)
            status = '✓' if abs(norm - 1.0) < 0.01 else '✗ NOT NORMALIZED'
            print(f"   Student {sid}: shape={enc.shape}, norm={norm:.6f} {status}")
    else:
        print("\n⚠️  Warning: Encoding file not found!")
    
    print("\n" + "=" * 70)
    print("🎉 Encoding rebuild complete! Restart your backend server.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
