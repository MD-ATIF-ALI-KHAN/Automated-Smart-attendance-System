"""
Teacher Verification API Endpoints
Endpoints for teacher-in-the-loop verification workflow
"""

from fastapi import APIRouter, HTTPException, Form
from typing import Optional, Dict
import os
import cv2

router = APIRouter()


@router.post("/verify-attendance")
async def verify_attendance_face(
    attendance_id: int = Form(...),
    face_index: int = Form(...),
    verified_student_id: Optional[str] = Form(None),
    action: str = Form(...)  # 'approve', 'reject', or 'unknown'
) -> Dict:
    """
    Submit teacher's verification decision for an uncertain face match
    
    Args:
        attendance_id: ID of the attendance record
        face_index: Index of the face in the pending verifications
        verified_student_id: Student ID confirmed by teacher (None if rejected/unknown)
        action: Verification action ('approve', 'reject', or 'unknown')
    
    Returns:
        Success status and whether encoding was added
    """
    print(f"\n{'='*80}")
    print(f"🔍 VERIFICATION REQUEST RECEIVED")
    print(f"{'='*80}")
    print(f"Attendance ID: {attendance_id}")
    print(f"Face Index: {face_index}")
    print(f"Verified Student ID: {verified_student_id}")
    print(f"Action: {action}")
    print(f"{'='*80}\n")
    
    try:
        from app.utils.verification_manager import verification_manager
        from app.services.attendance import AttendanceService
        from app.services.insightface_face_recognition import insightface_face_service
        
        # Validate action
        valid_actions = ['approve', 'reject', 'unknown']
        if action not in valid_actions:
            print(f"❌ Invalid action: {action}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {valid_actions}"
            )
        
        # Get pending verifications for this attendance
        print(f"📋 Fetching pending verifications for attendance {attendance_id}...")
        pending = verification_manager.get_pending_verifications(attendance_id)
        print(f"Found {len(pending)} pending verifications")
        
        if not pending:
            print(f"❌ No pending verifications found")
            raise HTTPException(
                status_code=404,
                detail="No pending verifications found for this attendance record"
            )
        
        # Find the specific verification
        print(f"🔍 Looking for verification with face_index={face_index}...")
        verification = next((v for v in pending if v['face_index'] == face_index), None)
        
        if not verification:
            print(f"❌ Verification not found for face_index={face_index}")
            print(f"Available face indices: {[v['face_index'] for v in pending]}")
            raise HTTPException(
                status_code=404,
                detail=f"No pending verification found for face index {face_index}"
            )
        
        verification_id = verification['id']
        print(f"✅ Found verification record ID: {verification_id}")
        
        # Update verification record
        print(f"💾 Updating verification record...")
        success = verification_manager.verify_face(
            verification_id=verification_id,
            verified_student_id=verified_student_id,
            action=action
        )
        
        if not success:
            print(f"❌ Failed to update verification record")
            raise HTTPException(
                status_code=500,
                detail="Failed to update verification record"
            )
        
        print(f"✅ Verification record updated successfully")
        
        
        encoding_added = False
        
        # If approved, add student to attendance
        if action == 'approve' and verified_student_id:
            print(f"\n👤 MARKING STUDENT AS PRESENT")
            print(f"Student ID: {verified_student_id}")
            print(f"Attendance ID: {attendance_id}")
            
            from app.services.attendance_helper import add_verified_student
            attendance_updated = await add_verified_student(attendance_id, verified_student_id)
            
            if attendance_updated:
                print(f"✅ Student {verified_student_id} marked as present")
            else:
                print(f"❌ Failed to mark student {verified_student_id} as present")
        
        # If approved, try to add encoding via continual learning
        if action == 'approve' and verified_student_id:
            print(f"\n🧠 ATTEMPTING TO ADD ENCODING (Continual Learning)")
            try:
                # Get face_crop_path and bbox from database
                from app.models.database import DatabaseManager
                db = DatabaseManager()
                query = """SELECT face_crop_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                          quality_score, suggested_similarity 
                          FROM attendance_verifications WHERE id = ?"""
                print(f"Querying database for verification ID: {verification_id}")
                result_db = db.execute_query(query, (verification_id,))
                
                if result_db and result_db[0]['face_crop_path']:
                    print(f"✅ Found face crop data in database")
                    face_crop_path = result_db[0]['face_crop_path']
                    print(f"Face crop path: {face_crop_path}")
                    
                    if os.path.exists(face_crop_path):
                        print(f"✅ Face crop file exists")
                        face_crop = cv2.imread(face_crop_path)
                        
                        
                        
                        # Get InsightFace service and add encoding
                        face_service = insightface_face_service
                        if not face_service.known_face_encodings:
                            await yolov8_service.load_encodings()
                        print(f"✅ YOLOv8 service initialized")
                        
                        # Extract bbox and other fields from database
                        db_record = result_db[0]
                        print(f"Database record fields: {list(db_record.keys())}")
                        
                        # Helper to safely convert bbox values (might be bytes from SQLite)
                        def safe_bbox_int(value):
                            if value is None:
                                return 0
                            if isinstance(value, bytes):
                                # Convert bytes to int (little-endian, unsigned)
                                result = int.from_bytes(value, byteorder='little', signed=False)
                                print(f"  Converted bytes {value} → {result}")
                                return result
                            try:
                                return int(value)
                            except (ValueError, TypeError):
                                print(f"  Failed to convert {value} to int, using 0")
                                return 0
                        
                        print(f"Converting bbox coordinates...")
                        bbox = (
                            safe_bbox_int(db_record['bbox_x1']),
                            safe_bbox_int(db_record['bbox_y1']),
                            safe_bbox_int(db_record['bbox_x2']),
                            safe_bbox_int(db_record['bbox_y2'])
                        )
                        print(f"✅ Bbox: {bbox}")
                        
                        
                        # Helper for safe float conversion
                        def safe_float(value):
                            if value is None:
                                return 0.0
                            if isinstance(value, bytes):
                                # Convert bytes to float via int first
                                try:
                                    int_val = int.from_bytes(value, byteorder='little', signed=False)
                                    return float(int_val)
                                except:
                                    return 0.0
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return 0.0
                        
                        confidence = safe_float(db_record['suggested_similarity'])
                        quality_score = safe_float(db_record['quality_score'])
                        print(f"Confidence: {confidence}, Quality: {quality_score}")
                        
                        # Add encoding from verified face
                        print(f"\n📸 Adding encoding to student {verified_student_id}...")
                        await yolov8_service.add_encoding_from_attendance(
                            student_id=verified_student_id,
                            face_crop=face_crop,
                            bbox=bbox,
                            confidence=confidence,
                            quality_score=quality_score,
                            force=True
                        )
                        
                        # Mark encoding as added
                        verification_manager.mark_encoding_added(verification_id)
                        encoding_added = True
                        
                        print(f"✅ Added encoding from verified face for student {verified_student_id}")
                    else:
                        print(f"⚠️  Face crop file not found: {face_crop_path}")
                else:
                    print(f"⚠️  No face_crop_path found in database for verification {verification_id}")
                    
            except Exception as e:
                print(f"⚠️  Failed to add encoding from verification: {e}")
                # Don't fail the verification if encoding addition fails
        
        # Update attendance record with verified student
        if action == 'approve' and verified_student_id:
            try:
                from app.services.attendance_helper import add_verified_student
                await add_verified_student(
                    attendance_id=attendance_id,
                    student_id=verified_student_id
                )
            except Exception as e:
                print(f"⚠️  Failed to update attendance record: {e}")
        
        return {
            "success": True,
            "message": f"Face verification {action}d successfully",
            "encoding_added": encoding_added,
            "action": action,
            "verified_student_id": verified_student_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending-verifications/{attendance_id}")
async def get_pending_verifications(attendance_id: int) -> Dict:
    """
    Get all pending verifications for an attendance record
    
    Args:
        attendance_id: ID of the attendance record
    
    Returns:
        List of pending verifications with face crops and candidates
    """
    try:
        from app.utils.verification_manager import verification_manager
        
        pending = verification_manager.get_pending_verifications(attendance_id)
        
        return {
            "success": True,
            "attendance_id": attendance_id,
            "pending_count": len(pending),
            "pending_verifications": pending
        }
        
    except Exception as e:
        print(f"Error getting pending verifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/verification-stats/{attendance_id}")
async def get_verification_stats(attendance_id: int) -> Dict:
    """
    Get verification statistics for an attendance record
    
    Args:
        attendance_id: ID of the attendance record
    
    Returns:
        Verification statistics
    """
    try:
        from app.utils.verification_manager import verification_manager
        
        stats = verification_manager.get_verification_stats(attendance_id)
        
        return {
            "success": True,
            "attendance_id": attendance_id,
            "stats": stats
        }
        
    except Exception as e:
        print(f"Error getting verification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
