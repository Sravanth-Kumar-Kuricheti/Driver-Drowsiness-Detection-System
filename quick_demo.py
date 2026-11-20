"""
Quick Demo Script - Driver Drowsiness Detection (Python 3.13 / Tasks API)
This script uses the modern MediaPipe Tasks API for landmark detection.
"""

import cv2
import numpy as np
import time
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from features import FeatureExtractor
from utils import preprocess_face_crop, extract_face_roi, draw_fps


def quick_demo():
    """
    Quick demonstration of the drowsiness detection pipeline
    Shows feature extraction in real-time even without trained model
    """
    
    print("\n" + "="*70)
    print("Quick Demo - Drowsiness Detection Feature Extraction (Tasks API)")
    print("="*70)
    print("\nThis demo shows real-time face detection and feature extraction.")
    print("Press 'Q' to quit\n")
    
    # Initialize MediaPipe Tasks FaceLandmarker
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        model_path_task = 'models/face_landmarker.task'
        if not os.path.exists(model_path_task):
            print(f"Error: {model_path_task} not found. Please run main system first or download model.")
            return
            
        base_options = python.BaseOptions(model_asset_path=model_path_task)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing MediaPipe Tasks: {e}")
        return
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.set_fps(30)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam opened successfully!")
    print("="*70 + "\n")
    
    start_time = time.time()
    frame_count = 0
    fps_list = []
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Prepare frame for MediaPipe Task
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            # Detect landmarks
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
                
                # Extract features
                timestamp = time.time() - start_time
                # Use the new task-compatible feature extraction method
                features = feature_extractor.extract_features_task(
                    face_landmarks, w, h, timestamp
                )
                
                # Get feature values (indices from features.py)
                ear_left = features[0]
                ear_right = features[1]
                ear_avg = (ear_left + ear_right) / 2
                mar = features[2]
                yaw = features[5]
                pitch = features[6]
                roll = features[7]
                drowsy_score = features[15]
                
                # Determine color based on drowsy score
                if drowsy_score > 0.6:
                    color = (0, 0, 255)  # Red
                    state_text = "DROWSY"
                elif drowsy_score > 0.3:
                    color = (0, 255, 255)  # Yellow
                    state_text = "WARNING"
                else:
                    color = (0, 255, 0)  # Green
                    state_text = "ALERT"
                
                # Draw face bounding box
                x_coords = [lm.x * w for lm in face_landmarks]
                y_coords = [lm.y * h for lm in face_landmarks]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                
                # Draw status bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                
                cv2.putText(frame, f"STATE: {state_text}", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Metrics
                y_offset = 120
                metrics = [
                    f"EAR: {ear_avg:.3f} {'[CLOSED]' if ear_avg < 0.20 else '[OPEN]'}",
                    f"MAR: {mar:.3f} {'[YAWNING]' if mar > 0.6 else ''}",
                    f"Head: Y={yaw:.1f} P={pitch:.1f} R={roll:.1f}",
                    f"Drowsy Score: {drowsy_score:.3f}",
                ]
                
                for i, metric in enumerate(metrics):
                    cv2.putText(frame, metric, (10, y_offset + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Extract face for display
                face_roi = extract_face_roi(frame, face_landmarks, w, h, padding=0.2)
                if face_roi is not None and face_roi.size > 0:
                    try:
                        face_processed = preprocess_face_crop(face_roi, target_size=(48, 48))
                        face_display = (face_processed[:, :, 0] * 255).astype(np.uint8)
                        face_display = cv2.resize(face_display, (96, 96))
                        frame[10:106, w-106:w-10] = cv2.cvtColor(face_display, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(frame, (w-106, 10), (w-10, 106), color, 2)
                    except:
                        pass
            
            else:
                cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 150, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate FPS
            loop_time = time.time() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0
            fps_list.append(current_fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)
            
            # Draw FPS
            frame = draw_fps(frame, avg_fps, position=(10, 30))
            
            # Display
            cv2.imshow('Quick Demo - Feature Extraction', frame)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if 'detector' in locals():
            detector.close()
        
        print("\nDemo Summary")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {np.mean(fps_list):.1f}")


if __name__ == "__main__":
    quick_demo()
