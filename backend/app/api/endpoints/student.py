from fastapi import APIRouter, HTTPException, Depends, status, File, UploadFile, Form
from typing import List, Optional
import shutil
import os
import logging
from datetime import datetime
from app.models.database import DatabaseManager
from app.services.class_service import ClassService
from app.services.face_recognition import FaceRecognitionService
from app.api.deps import get_current_student
from app.models.auth_schemas import UserResponse, ClassResponse

logger = logging.getLogger(__name__)

router = APIRouter()
face_service = FaceRecognitionService()

@router.get("/profile", response_model=UserResponse)
async def get_student_profile(current_user: dict = Depends(get_current_student)):
    """Get student profile"""
    try:
        user = DatabaseManager.execute_query(
            "SELECT * FROM users WHERE id = ?",
            (current_user['id'],)
        )
        
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        user_data = user[0]
        return UserResponse(
            id=user_data['id'],
            email=user_data['email'],
            full_name=user_data['full_name'],
            role=user_data['role'],
            is_active=user_data['is_active'],
            created_at=user_data['created_at']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.put("/profile")
async def update_student_profile(
    full_name: Optional[str] = None,
    current_user: dict = Depends(get_current_student)
):
    """Update student profile"""
    try:
        if full_name:
            DatabaseManager.execute_update(
                "UPDATE users SET full_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (full_name, current_user['id'])
            )
            
            # Also update student name
            DatabaseManager.execute_update(
                "UPDATE students SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (full_name, current_user['id'])
            )
        
        return {"message": "Profile updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/classes")
async def get_student_classes(current_user: dict = Depends(get_current_student)):
    """Get all classes the student is enrolled in"""
    try:
        # Get student record to get student_id
        student = DatabaseManager.execute_query(
            "SELECT student_id FROM students WHERE user_id = ?",
            (current_user['id'],)
        )
        
        if not student:
            return []  # Student hasn't uploaded video yet, no enrollments
        
        student_id = student[0]['student_id']
        
        # Get enrolled classes
        classes = DatabaseManager.execute_query(
            """SELECT c.id, c.class_name, c.description, c.created_at, ce.enrolled_at,
                      u.full_name as teacher_name
               FROM class_enrollments ce
               JOIN classes c ON ce.class_id = c.id
               JOIN users u ON c.teacher_id = u.id
               WHERE ce.student_id = ?
               ORDER BY ce.enrolled_at DESC""",
            (student_id,)
        )
        
        return classes
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/attendance")
async def get_student_attendance(
    class_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_student)
):
    """Get student's personal attendance records"""
    try:
        # Get student record
        student = DatabaseManager.execute_query(
            "SELECT student_id FROM students WHERE user_id = ?",
            (current_user['id'],)
        )
        
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student record not found")
        
        student_id = student[0]['student_id']
        
        # Build query
        query = """
            SELECT ar.id, ar.class_name, ar.date, ar.created_at,
                   sa.status, sa.confidence
            FROM attendance_records ar
            JOIN student_attendance sa ON ar.id = sa.attendance_record_id
            WHERE sa.student_id = ?
        """
        params = [student_id]
        
        if class_id:
            query += " AND ar.class_id = ?"
            params.append(class_id)
        
        if start_date:
            query += " AND ar.date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND ar.date <= ?"
            params.append(end_date)
        
        query += " ORDER BY ar.date DESC, ar.created_at DESC"
        
        records = DatabaseManager.execute_query(query, tuple(params))
        
        # Calculate statistics
        total_sessions = len(records)
        present_count = sum(1 for r in records if r['status'] == 'present')
        attendance_percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0.0
        
        return {
            "records": [
                {
                    "id": r['id'],
                    "class_name": r['class_name'],
                    "date": r['date'],
                    "timestamp": r['created_at'],
                    "status": r['status'],
                    "confidence": r.get('confidence', 0.0)
                }
                for r in records
            ],
            "statistics": {
                "total_sessions": total_sessions,
                "present_count": present_count,
                "absent_count": total_sessions - present_count,
                "attendance_percentage": round(attendance_percentage, 2)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/video")
async def upload_student_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_student)
):
    """Upload face recognition video for student"""
    try:
        # Get student record
        student = DatabaseManager.execute_query(
            "SELECT student_id, name, class_name FROM students WHERE user_id = ?",
            (current_user['id'],)
        )
        
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student record not found")
        
        student_id = student[0]['student_id']
        student_name = student[0]['name']
        class_name = student[0].get('class_name', 'default')
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a video")
        
        # Create student directory
        student_dir = f"data/student_videos/{student_id}"
        os.makedirs(student_dir, exist_ok=True)
        
        # Save video file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{student_id}_{timestamp}.mp4"
        video_path = os.path.join(student_dir, video_filename)
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process video: extract frames and generate face encodings
        from app.services.video_processing import VideoProcessingService
        video_service = VideoProcessingService()
        
        result = video_service.process_student_video(
            video_path=video_path,
            student_id=student_id,
            student_name=student_name,
            class_name=class_name
        )
        
        # Update student record with video path
        DatabaseManager.execute_update(
            "UPDATE students SET image_path = ?, updated_at = CURRENT_TIMESTAMP WHERE student_id = ?",
            (video_path, student_id)
        )
        
        # Generate face encodings from extracted frames
        faces_detected = 0
        encodings_generated = False
        frames_processed = 0
        frames_rejected = 0
        
        if result.get('success') and result.get('frame_paths'):
            try:
                # Process each frame to detect and encode faces
                from app.services.insightface_face_recognition import InsightFaceFaceRecognitionService
                face_service = InsightFaceFaceRecognitionService()
                
                logger.info(f"Processing {len(result['frame_paths'])} frames for student {student_id}")
                
                # Generate encodings for this student from all frames
                # Try each frame until we get a successful encoding
                for frame_path in result['frame_paths']:
                    if os.path.exists(frame_path):
                        frames_processed += 1
                        # Generate encoding for this frame with relaxed quality checks for video frames
                        encoding = await yolo_service.generate_encoding(frame_path, student_id, is_video_frame=True)
                        if encoding is not None:
                            faces_detected += 1
                            encodings_generated = True
                            logger.info(f"✅ Successfully generated encoding from frame {frames_processed}/{len(result['frame_paths'])}")
                            # We only need one good encoding per student, so break after first success
                            break
                        else:
                            frames_rejected += 1
                            logger.debug(f"Frame {frames_processed} rejected, trying next frame...")
                
                if encodings_generated:
                    logger.info(f"✅ Generated face encoding for student {student_name} ({student_id}) after processing {frames_processed} frames")
                else:
                    logger.warning(f"❌ No faces detected in any of the {frames_processed} frames for student {student_id}")
                    logger.warning(f"   All {frames_rejected} frames were rejected by quality checks")
                
            except Exception as e:
                logger.error(f"Error generating face encodings: {e}")
                import traceback
                traceback.print_exc()
        
        # Reload face encodings in the face recognition service
        await face_service.load_encodings()
        
        return {
            "message": "Video processed successfully",
            "student_id": student_id,
            "video_path": video_path,
            "frames_extracted": result.get('frames_extracted', 0),
            "faces_detected": faces_detected,
            "encodings_generated": encodings_generated
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/video")
async def delete_student_video(current_user: dict = Depends(get_current_student)):
    """Delete face recognition video and encodings"""
    try:
        # Get student record
        student = DatabaseManager.execute_query(
            "SELECT student_id, image_path FROM students WHERE user_id = ?",
            (current_user['id'],)
        )
        
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student record not found")
        
        student_id = student[0]['student_id']
        image_path = student[0].get('image_path')
        
        # Delete student directory
        student_dir = f"data/student_images/{student_id}"
        if os.path.exists(student_dir):
            shutil.rmtree(student_dir)
        
        # Clear image path in database
        DatabaseManager.execute_update(
            "UPDATE students SET image_path = NULL, updated_at = CURRENT_TIMESTAMP WHERE student_id = ?",
            (student_id,)
        )
        
        # Remove face encodings
        await face_service.remove_encoding(student_id)
        
        return {"message": "Video and encodings deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/join-class")
async def join_class(
    data: dict,
    current_user: dict = Depends(get_current_student)
):
    """Allow student to join additional classes using invite code"""
    try:
        invite_code = data.get('invite_code', '').strip()
        
        if not invite_code:
            raise HTTPException(status_code=400, detail="Invite code is required")
        
        # Get the class by invite code
        class_query = """
            SELECT id, class_name, teacher_id 
            FROM classes 
            WHERE invite_code = ? AND (invite_expires_at IS NULL OR invite_expires_at > CURRENT_TIMESTAMP)
        """
        class_result = DatabaseManager.execute_query(class_query, (invite_code,))
        
        if not class_result:
            raise HTTPException(status_code=404, detail="Invalid or expired invite code")
        
        class_data = class_result[0]
        class_id = class_data['id']
        class_name = class_data['class_name']
        
        # Check if student already has a profile
        existing_student = DatabaseManager.execute_query(
            "SELECT * FROM students WHERE user_id = ?",
            (current_user['id'],)
        )
        
        if not existing_student:
            raise HTTPException(
                status_code=400, 
                detail="Please upload your face recognition video first before joining a class"
            )
        
        student_data = existing_student[0]
        student_id = student_data['student_id']
        
        # Check if already enrolled in this class
        existing_enrollment = DatabaseManager.execute_query(
            "SELECT * FROM class_enrollments WHERE student_id = ? AND class_id = ?",
            (student_id, class_id)
        )
        
        if existing_enrollment:
            raise HTTPException(status_code=400, detail=f"You are already enrolled in {class_name}")
        
        # Add enrollment
        DatabaseManager.execute_update(
            """INSERT INTO class_enrollments (student_id, class_id, enrolled_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (student_id, class_id)
        )
        
        logger.info(f"Student {student_id} joined class {class_name} (ID: {class_id})")
        
        return {
            "message": "Successfully joined class",
            "class_name": class_name,
            "class_id": class_id,
            "student_id": student_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining class: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/leave-class/{class_id}")
async def leave_class(
    class_id: int,
    current_user: dict = Depends(get_current_student)
):
    """Allow student to leave a class"""
    try:
        # Get student record
        student = DatabaseManager.execute_query(
            "SELECT student_id FROM students WHERE user_id = ?",
            (current_user['id'],)
        )
        
        if not student:
            raise HTTPException(status_code=404, detail="Student record not found")
        
        student_id = student[0]['student_id']
        
        # Check if enrolled in this class
        enrollment = DatabaseManager.execute_query(
            "SELECT * FROM class_enrollments WHERE student_id = ? AND class_id = ?",
            (student_id, class_id)
        )
        
        if not enrollment:
            raise HTTPException(status_code=404, detail="You are not enrolled in this class")
        
        # Remove enrollment
        DatabaseManager.execute_update(
            "DELETE FROM class_enrollments WHERE student_id = ? AND class_id = ?",
            (student_id, class_id)
        )
        
        logger.info(f"Student {student_id} left class {class_id}")
        
        return {"message": "Successfully left class"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leaving class: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report")
async def get_student_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_student)
):
    """Generate attendance report for the logged-in student"""
    try:
        # Get student record
        student_query = """
            SELECT student_id, name, class_name 
            FROM students 
            WHERE user_id = ?
        """
        student_result = DatabaseManager.execute_query(student_query, (current_user['id'],))
        
        if not student_result:
            raise HTTPException(status_code=404, detail="Student record not found")
        
        student = student_result[0]
        student_id = student['student_id']
        
        # Build query for attendance records
        query = """
            SELECT 
                ar.date,
                ar.class_name,
                sa.status,
                sa.confidence,
                ar.created_at
            FROM student_attendance sa
            JOIN attendance_records ar ON sa.attendance_record_id = ar.id
            WHERE sa.student_id = ?
        """
        params = [student_id]
        
        if start_date:
            query += " AND ar.date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND ar.date <= ?"
            params.append(end_date)
        
        query += " ORDER BY ar.date DESC"
        
        records = DatabaseManager.execute_query(query, tuple(params))
        
        # Calculate statistics
        total_sessions = len(records)
        present_count = sum(1 for r in records if r['status'] == 'present')
        absent_count = total_sessions - present_count
        attendance_percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0
        
        return {
            "student_name": student['name'],
            "class_name": student['class_name'],
            "total_sessions": total_sessions,
            "present_count": present_count,
            "absent_count": absent_count,
            "attendance_percentage": attendance_percentage,
            "records": records,
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating student report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/export")
async def export_student_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_student)
):
    """Export student attendance report to Excel"""
    try:
        import pandas as pd
        
        # Get student record
        student_query = """
            SELECT student_id, name, class_name 
            FROM students 
            WHERE user_id = ?
        """
        student_result = DatabaseManager.execute_query(student_query, (current_user['id'],))
        
        if not student_result:
            raise HTTPException(status_code=404, detail="Student record not found")
        
        student = student_result[0]
        student_id = student['student_id']
        
        # Build query for attendance records
        query = """
            SELECT 
                ar.date,
                ar.class_name,
                sa.status,
                sa.confidence,
                ar.created_at
            FROM student_attendance sa
            JOIN attendance_records ar ON sa.attendance_record_id = ar.id
            WHERE sa.student_id = ?
        """
        params = [student_id]
        
        if start_date:
            query += " AND ar.date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND ar.date <= ?"
            params.append(end_date)
        
        query += " ORDER BY ar.date DESC"
        
        records = DatabaseManager.execute_query(query, tuple(params))
        
        # Calculate statistics
        total_sessions = len(records)
        present_count = sum(1 for r in records if r['status'] == 'present')
        attendance_percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0
        
        # Create Excel file
        os.makedirs("data/exports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"student_report_{student['name'].replace(' ', '_')}_{timestamp}.xlsx"
        filepath = f"data/exports/{filename}"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = [
                ['Student Name', student['name']],
                ['Class', student['class_name']],
                ['Total Sessions', total_sessions],
                ['Present', present_count],
                ['Absent', total_sessions - present_count],
                ['Attendance Percentage', f"{attendance_percentage:.2f}%"],
                ['Start Date', start_date or 'All'],
                ['End Date', end_date or 'All']
            ]
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed records sheet
            if records:
                records_df = pd.DataFrame(records)
                records_df.to_excel(writer, sheet_name='Attendance Records', index=False)
        
        logger.info(f"Exported student report to {filepath}")
        return {"download_url": f"/static/exports/{filename}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting student report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
