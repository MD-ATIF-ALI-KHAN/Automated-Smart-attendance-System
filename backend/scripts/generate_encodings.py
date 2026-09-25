"""
Process existing student videos and generate face encodings
This script extracts frames from videos and generates face encodings
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.video_processing import VideoProcessingService
from app.services.insightface_face_recognition import InsightFaceFaceRecognitionService
from app.models.database import DatabaseManager
import asyncio

async def process_existing_videos():
    """Process all student videos and generate encodings"""
    
    # Get students with videos
    students = DatabaseManager.execute_query(
        "SELECT student_id, name, class_name, image_path FROM students WHERE image_path IS NOT NULL"
    )
    
    if not students:
        print("❌ No students with uploaded videos found")
        return
    
    print(f"Found {len(students)} students with videos")
    print("=" * 70)
    
    video_service = VideoProcessingService()
    face_service = InsightFaceFaceRecognitionService()
    
    for student in students:
        student_id = student['student_id']
        name = student['name']
        class_name = student.get('class_name', 'default')
        video_path = student['image_path']
        
        print(f"\n📹 Processing: {name} (Roll: {student_id}, Class: {class_name})")
        print(f"   Video: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"   ❌ Video file not found!")
            continue
        
        try:
            # Step 1: Extract frames from video
            print(f"   🎬 Extracting frames...")
            result = video_service.process_student_video(
                video_path=video_path,
                student_id=student_id,
                student_name=name,
                class_name=class_name
            )
            
            if not result.get('success'):
                print(f"   ❌ Failed to extract frames")
                continue
            
            frame_paths = result.get('frame_paths', [])
            print(f"   ✅ Extracted {len(frame_paths)} frames")
            
            # Step 2: Generate encodings from frames
            print(f"   🔍 Generating face encodings...")
            faces_detected = 0
            encodings_generated = False
            
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    encoding = await face_service.generate_encoding(frame_path, student_id)
                    if encoding is not None:
                        faces_detected += 1
                        encodings_generated = True
            
            if encodings_generated:
                print(f"   ✅ Generated {faces_detected} face encodings!")
            else:
                print(f"   ⚠️  No faces detected in frames")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("\nFace encodings have been generated for all students.")
    print("You can now mark attendance using classroom photos.")

if __name__ == "__main__":
    asyncio.run(process_existing_videos())
