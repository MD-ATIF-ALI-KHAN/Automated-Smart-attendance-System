import asyncio
import sys
import os
import cv2
import sqlite3
import numpy as np

# Add backend to path
sys.path.append(os.getcwd())

from app.models.database import DatabaseManager
from app.services.insightface_face_recognition import InsightFaceFaceRecognitionService
from app.utils.verification_manager import VerificationManager
from app.services.attendance_helper import add_verified_student

async def test_verification_flow(verification_id, attendance_id, student_id):
    print(f"\n{'='*60}")
    print(f"🧪 TESTING VERIFICATION FLOW")
    print(f"{'='*60}")
    print(f"Verification ID: {verification_id}")
    print(f"Attendance ID: {attendance_id}")
    print(f"Student ID: {student_id}")

    # 1. Check initial state
    print(f"\n📊 Checking initial state...")
    db = DatabaseManager()
    
    # Check verification record
    v_query = "SELECT * FROM attendance_verifications WHERE id = ?"
    v_record = db.execute_query(v_query, (verification_id,))
    if not v_record:
        print("❌ Verification record not found!")
        return
    print(f"Verification record: {v_record[0]}")

    # Check attendance record (student_attendance)
    sa_query = "SELECT * FROM student_attendance WHERE attendance_record_id = ? AND student_id = ?"
    sa_record = db.execute_query(sa_query, (attendance_id, student_id))
    print(f"Student attendance record: {sa_record[0] if sa_record else 'None'}")

    # 2. Simulate add_verified_student
    print(f"\n👤 Simulating add_verified_student...")
    success = await add_verified_student(attendance_id, student_id)
    print(f"add_verified_student result: {success}")

    # Verify database update
    sa_record_after = db.execute_query(sa_query, (attendance_id, student_id))
    print(f"Student attendance record after: {sa_record_after[0] if sa_record_after else 'None'}")
    if sa_record_after and sa_record_after[0]['status'] == 'present':
        print("✅ Student correctly marked as present")
    else:
        print("❌ Student NOT marked as present")

    # 3. Simulate encoding addition (The core issue)
    print(f"\n🧠 Simulating encoding addition...")
    
    # helper functions from verification.py
    def safe_bbox_int(value):
        if value is None:
            return 0
        if isinstance(value, bytes):
            return int.from_bytes(value, byteorder='little', signed=False)
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def safe_float(value):
        if value is None:
            return 0.0
        if isinstance(value, bytes):
             try:
                int_val = int.from_bytes(value, byteorder='little', signed=False)
                return float(int_val)
             except:
                return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    record = v_record[0]
    
    # Parse values
    bbox = (
        safe_bbox_int(record['bbox_x1']),
        safe_bbox_int(record['bbox_y1']),
        safe_bbox_int(record['bbox_x2']),
        safe_bbox_int(record['bbox_y2'])
    )
    print(f"Parsed bbox: {bbox} (Types: {[type(x) for x in bbox]})")
    
    confidence = safe_float(record['suggested_similarity'])
    quality = safe_float(record['quality_score'])
    print(f"Parsed confidence: {confidence}, quality: {quality}")

    # Load face crop
    face_crop_path = record['face_crop_path']
    if face_crop_path and os.path.exists(face_crop_path):
        print(f"Loading face crop from: {face_crop_path}")
        face_crop = cv2.imread(face_crop_path)
        print(f"Face crop shape: {face_crop.shape}")
        
        # Initialize service
        face_service = InsightFaceFaceRecognitionService()
        
        # Attempt to add encoding
        try:
            await yolov8_service.add_encoding_from_attendance(
                student_id=student_id,
                face_crop=face_crop,
                bbox=bbox,
                confidence=confidence,
                quality_score=quality
            )
            print("✅ add_encoding_from_attendance completed without error")
            
            # Mark encoding as added
            vm = VerificationManager()
            vm.mark_encoding_added(verification_id)
            print("✅ Marked encoding as added in database")
            
        except Exception as e:
            print(f"❌ Error adding encoding: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Face crop not found at {face_crop_path}")

    # 4. Final Verification
    print(f"\n🏁 Final Check...")
    v_record_final = db.execute_query(v_query, (verification_id,))
    print(f"Verification record final: {v_record_final[0]}")
    if v_record_final[0]['encoding_added']:
        print("✅ Encoding addition flag verified in DB")
    else:
        print("❌ Encoding addition flag NOT set")

if __name__ == "__main__":
    # verification_id=111, attendance_id=90, student_id='118' (Atif)
    asyncio.run(test_verification_flow(111, 90, '118'))
