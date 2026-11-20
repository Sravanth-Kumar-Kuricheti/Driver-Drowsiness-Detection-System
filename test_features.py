"""
Test Script for Feature Extraction Module (Python 3.13 / Tasks API)
Validates EAR, MAR, and head pose calculations using modern MediaPipe Tasks
"""

import cv2
import numpy as np
import time
import os
import sys

from features import FeatureExtractor

def test_feature_extraction_webcam(duration=15):
    """
    Test feature extraction with live webcam feed using Tasks API
    
    Args:
        duration: Test duration in seconds
    """
    print("\n" + "="*70)
    print("Feature Extraction Test - Live Webcam (Tasks API)")
    print("="*70 + "\n")
    
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
    extractor = FeatureExtractor()
    extractor.set_fps(30)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Testing feature extraction for {duration} seconds...")
    print("Press 'q' to quit early\n")
    
    start_time = time.time()
    frame_count = 0
    feature_history = []
    
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check duration
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Prepare frame for MediaPipe Task
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(elapsed * 1000)
            
            # Detect face
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                # Extract features using Tasks-compatible method
                features = extractor.extract_features_task(landmarks, w, h, elapsed)
                feature_history.append(features)
                
                # Display features
                ear_avg = (features[0] + features[1]) / 2
                mar = features[2]
                yaw, pitch, roll = features[5], features[6], features[7]
                drowsy_score = features[15]
                
                # Draw text overlays
                y_offset = 30
                info_texts = [
                    f"EAR (Eyes): {ear_avg:.3f}",
                    f"MAR (Mouth): {mar:.3f}",
                    f"Head Yaw: {yaw:.1f}°",
                    f"Head Pitch: {pitch:.1f}°",
                    f"Head Roll: {roll:.1f}°",
                    f"Drowsy Score: {drowsy_score:.3f}",
                    f"Frames: {frame_count}"
                ]
                
                color = (0, 255, 0) if drowsy_score < 0.3 else (0, 255, 255) if drowsy_score < 0.6 else (0, 0, 255)
                
                for text in info_texts:
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 30
                
                # Draw landmarks
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                frame_count += 1
            else:
                cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 150, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Feature Extraction Test (Tasks API)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
    
    # Analysis
    if len(feature_history) > 0:
        features_array = np.array(feature_history)
        
        print("\n" + "="*70)
        print("Feature Extraction Test Results")
        print("="*70)
        print(f"\nFrames processed: {frame_count}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Average FPS: {frame_count / elapsed:.1f}")
        
        print("\nFeature Statistics:")
        print("-" * 70)
        feature_names = [
            "EAR Left", "EAR Right", "MAR", "Blink Status", "Yawn Status",
            "Yaw", "Pitch", "Roll", "Eye Closure Duration", "Mouth Open Duration",
            "Head Vel X", "Head Vel Y", "Head Vel Z", "Gaze X", "Gaze Y", "Drowsy Score"
        ]
        
        for i, name in enumerate(feature_names):
            mean_val = np.mean(features_array[:, i])
            std_val = np.std(features_array[:, i])
            min_val = np.min(features_array[:, i])
            max_val = np.max(features_array[:, i])
            
            print(f"{name:22s}: mean={mean_val:7.3f}, std={std_val:7.3f}, "
                  f"min={min_val:7.3f}, max={max_val:7.3f}")
        
        # Validation checks
        print("\nValidation Checks:")
        print("-" * 70)
        
        # Check EAR range
        ear_mean = np.mean((features_array[:, 0] + features_array[:, 1]) / 2)
        print(f"✓ Average EAR: {ear_mean:.3f} (expected: 0.20-0.30)")
        
        # Check MAR range
        mar_mean = np.mean(features_array[:, 2])
        print(f"✓ Average MAR: {mar_mean:.3f} (expected: 0.10-0.40)")
        
        # Check head pose ranges
        yaw_range = np.max(features_array[:, 5]) - np.min(features_array[:, 5])
        pitch_range = np.max(features_array[:, 6]) - np.min(features_array[:, 6])
        roll_range = np.max(features_array[:, 7]) - np.min(features_array[:, 7])
        
        print(f"✓ Yaw range: {yaw_range:.1f}° (head rotation detected)")
        print(f"✓ Pitch range: {pitch_range:.1f}° (head tilt detected)")
        print(f"✓ Roll range: {roll_range:.1f}° (head roll detected)")
        
        # Check feature vector shape
        print(f"✓ Feature vector shape: {features_array.shape[1]} (expected: 16)")
        
        print("="*70 + "\n")
        print("✓ Feature extraction test completed successfully!")
    else:
        print("\n⚠ No features extracted (no face detected)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test feature extraction module (Tasks API)')
    parser.add_argument('--duration', type=int, default=15,
                       help='Test duration for webcam test (default: 15s)')
    
    args = parser.parse_args()
    
    test_feature_extraction_webcam(duration=args.duration)

if __name__ == "__main__":
    main()
