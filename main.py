"""
Real-Time Driver Drowsiness Detection System
Main entry point for webcam-based drowsiness monitoring
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import os
import sys

from features import FeatureExtractor
from alerts import AlertSystem
from utils import (
    SequenceBuffer, VideoRecorder, preprocess_face_crop,
    extract_face_roi, draw_fps, create_output_dirs,
    enhance_low_light_frame
)

# Lazy import model to avoid protobuf conflict
# from model import load_pretrained_model 

class DrowsinessDetectionSystem:
    """
    Complete real-time drowsiness detection system
    Integrates MediaPipe, feature extraction, CNN-LSTM model, and alerts
    """
    
    def __init__(self, model_path='models/drowsiness_detector.h5',
                 camera_id=0, enable_audio=True, enable_recording=True,
                 output_dir='output', debug=False,
                 enable_calibration=True, enable_low_light=True):
        """
        Initialize drowsiness detection system
        
        Args:
            model_path: Path to trained model
            camera_id: Camera device ID or video file path
            enable_audio: Enable audio alerts
            enable_recording: Enable drowsy event recording
            output_dir: Output directory for logs and recordings
            debug: Show debug information
            enable_calibration: Run 5-sec personal baseline calibration
            enable_low_light: Dynamically enhance low-light cabin frames
        """
        self.camera_id = camera_id
        self.debug = debug
        self.enable_calibration = enable_calibration
        self.enable_low_light = enable_low_light
        
        # Create output directories
        self.output_dirs = create_output_dirs(output_dir)
        
        print("\n" + "="*70)
        print("Driver Drowsiness Detection System")
        print("="*70 + "\n")
        
        # Initialize MediaPipe Tasks FaceLandmarker (Critical for Python 3.13 support)
        print("[1/5] Initializing MediaPipe FaceLandmarker...")
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            model_path_task = 'models/face_landmarker.task'
            if not os.path.exists(model_path_task):
                print(f"      ⚠ Error: {model_path_task} not found. Please ensure it was downloaded.")
                sys.exit(1)
                
            base_options = python.BaseOptions(model_asset_path=model_path_task)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO, # optimized for video
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("      ✓ MediaPipe Task Detector initialized")
        except Exception as e:
            print(f"      ⚠ Critical Error initializing MediaPipe: {e}")
            sys.exit(1)
        
        # Initialize feature extractor
        print("[2/5] Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor()
        print("      ✓ Feature extractor ready")
        
        # Load drowsiness detection model (Lazy Load)
        print(f"[3/5] Loading model from {model_path}...")
        try:
            from model import load_pretrained_model
            self.predictor = load_pretrained_model(model_path)
            if self.predictor.model is None:
                print("      ⚠ Warning: Model not loaded. System will run in feature extraction mode only.")
                self.model_loaded = False
            else:
                print("      ✓ Model loaded successfully")
                self.model_loaded = True
        except Exception as e:
            print(f"      ⚠ Error loading model: {e}")
            print("      ⚠ System will run in HEURISTIC MODE (no deep learning)")
            self.model_loaded = False
            self.predictor = None
        
        # Initialize alert system
        print("[4/5] Initializing alert system...")
        self.alert_system = AlertSystem(
            log_dir=self.output_dirs['logs'],
            enable_audio=enable_audio,
            enable_voice=enable_audio
        )
        print("      ✓ Alert system ready")
        
        # Initialize video recorder
        self.enable_recording = enable_recording
        if enable_recording:
            print("[5/5] Initializing video recorder...")
            self.video_recorder = VideoRecorder(
                output_dir=self.output_dirs['recordings'],
                fps=30,
                pre_event_seconds=5
            )
            print("      ✓ Video recorder ready")
        else:
            self.video_recorder = None
            print("[5/5] Video recording disabled")
        
        # Initialize sequence buffer for temporal analysis
        self.face_buffer = SequenceBuffer(max_length=16)
        
        # Start personal calibration if enabled
        if self.enable_calibration:
            self.feature_extractor.start_calibration(duration=5.0)
        
        # Performance tracking
        self.fps_list = []
        self.frame_count = 0
        self.last_prediction = None
        self.last_features = None
        
        print("\n" + "="*70)
        print("System initialized successfully!")
        print("="*70 + "\n")
    
    def process_frame(self, frame, timestamp):
        """
        Process single video frame for drowsiness detection
        
        Args:
            frame: BGR video frame
            timestamp: Frame timestamp in seconds
        
        Returns:
            tuple: (processed_frame, prediction_result)
        """
        # Low-light cabin enhancement (CLAHE on L-channel)
        if self.enable_low_light:
            frame = enhance_low_light_frame(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Prepare frame for MediaPipe Task
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # MediaPipe Task expects timestamp in MS
        # Use the provided timestamp argument, converted to milliseconds
        timestamp_ms = int(timestamp * 1000)
        
        # Detect landmarks
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        # Process results
        if not detection_result.face_landmarks:
            # No face detected
            cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 150, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.face_buffer.clear()
            return frame, None
        
        # Use first face detected
        face_landmarks_list = detection_result.face_landmarks[0]
        
        # Extract features using 3.13-compatible wrapper
        features = self.feature_extractor.extract_features_task(
            face_landmarks_list, w, h, timestamp
        )
        self.last_features = features
        
        # Update personal calibration if active
        if self.enable_calibration and self.feature_extractor.is_calibrating:
            ear_avg = (features[0] + features[1]) / 2.0
            mar = features[2]
            self.feature_extractor.update_calibration(ear_avg, mar, timestamp)
        
        # Extract face ROI for CNN input
        face_roi = extract_face_roi(frame, face_landmarks_list, w, h, padding=0.2)
        
        if face_roi is not None:
            # Preprocess face for model
            face_processed = preprocess_face_crop(face_roi, target_size=(48, 48))
            
            # Add to sequence buffer
            self.face_buffer.add(face_processed)
        
        # Run model prediction if buffer is full
        prediction_result = None
        
        if self.model_loaded and self.face_buffer.is_full():
            current_sequence = self.face_buffer.get_sequence()
            
            try:
                # Get prediction from CNN-LSTM
                prediction_result = self.predictor.predict(current_sequence)
                
                # Safeguard: Heuristic override for obvious "Alert" behaviors
                # This fixes over-sensitivity issues reported by users
                ear_avg = (features[0] + features[1]) / 2.0
                mar = features[2]
                pitch = features[6]
                
                # If eyes are open, mouth closed, and head upright, force ALERT state
                is_heuristically_alert = (ear_avg > 0.22) and (mar < 0.45) and (pitch < 20)
                
                if is_heuristically_alert and prediction_result['state'] != 0:
                    prediction_result['state'] = 0
                    prediction_result['state_name'] = "ALERT"
                    prediction_result['confidence'] = 1.0
                
                self.last_prediction = prediction_result
                
                # Update alert system
                self.alert_system.update_state(prediction_result, features)
                
                # Handle video recording
                if self.enable_recording and self.video_recorder:
                    if prediction_result['state'] == AlertSystem.STATE_DROWSY:
                        if not self.video_recorder.is_recording:
                            self.video_recorder.start_recording(frame)
                        else:
                            self.video_recorder.record_frame(frame)
                    elif self.video_recorder.is_recording:
                        # Continue recording for a few more seconds after drowsy event
                        self.video_recorder.record_frame(frame)
                        if prediction_result['state'] == AlertSystem.STATE_ALERT:
                            # Stop recording after returning to alert state
                            pass  # Let it auto-stop after max duration
            
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction_result = None
        
        elif not self.model_loaded:
            # Run in Heuristic Mode (MediaPipe Eye, Mouth, and Head Pose features)
            drowsiness_score = float(features[15])
            if drowsiness_score < 0.35:
                state = AlertSystem.STATE_ALERT
                state_name = "ALERT"
            elif drowsiness_score < 0.65:
                state = AlertSystem.STATE_WARNING
                state_name = "WARNING"
            else:
                state = AlertSystem.STATE_DROWSY
                state_name = "DROWSY"
            
            prediction_result = {
                'state': state,
                'state_name': state_name,
                'confidence': drowsiness_score,
                'probabilities': [max(0.0, 1.0 - drowsiness_score), 0.5 * drowsiness_score, drowsiness_score]
            }
            self.last_prediction = prediction_result
            
            # Update alert system
            self.alert_system.update_state(prediction_result, features)
            
            # Handle video recording
            if self.enable_recording and self.video_recorder:
                if state == AlertSystem.STATE_DROWSY:
                    if not self.video_recorder.is_recording:
                        self.video_recorder.start_recording(frame)
                    else:
                        self.video_recorder.record_frame(frame)
                elif self.video_recorder.is_recording:
                    self.video_recorder.record_frame(frame)
        
        # Add frame to pre-event buffer (even if not recording)
        if self.enable_recording and self.video_recorder:
            self.video_recorder.add_frame(frame)
        
        # Draw visualizations
        frame = self._draw_visualizations(frame, face_landmarks_list, features, 
                                          prediction_result, w, h, timestamp)
        
        return frame, prediction_result
    
    def _draw_visualizations(self, frame, face_landmarks, features, 
                            prediction_result, width, height, timestamp=0.0):
        """
        Draw all visualizations on frame
        
        Args:
            frame: Video frame
            face_landmarks: MediaPipe landmarks
            features: Extracted features
            prediction_result: Model prediction
            width: Frame width
            height: Frame height
            timestamp: Frame timestamp in seconds
        
        Returns:
            np.array: Frame with visualizations
        """
        # Calculate face bounding box
        x_coords = [lm.x * width for lm in face_landmarks]
        y_coords = [lm.y * height for lm in face_landmarks]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Draw landmarks if in debug mode
        if self.debug:
            for lm in face_landmarks:
                x, y = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Calibration & PERCLOS metadata
        cal_start = self.feature_extractor.calibration_start_time or timestamp
        elapsed_cal = max(0.0, timestamp - cal_start)
        cal_info = {
            'is_calibrating': self.feature_extractor.is_calibrating,
            'is_calibrated': self.feature_extractor.is_calibrated,
            'baseline_ear': self.feature_extractor.baseline_ear,
            'baseline_mar': self.feature_extractor.baseline_mar,
            'ear_threshold': self.feature_extractor.EAR_THRESHOLD,
            'mar_threshold': self.feature_extractor.MAR_THRESHOLD,
            'remaining': max(0.0, self.feature_extractor.calibration_duration - elapsed_cal),
            'progress': min(1.0, elapsed_cal / self.feature_extractor.calibration_duration)
        }
        perclos = self.feature_extractor.current_perclos
        
        # Draw alert overlays and telemetry HUD
        if prediction_result is not None:
            frame = self.alert_system.draw_visual_alerts(
                frame, prediction_result, features, face_bbox,
                calibration_info=cal_info, perclos=perclos
            )
        else:
            # Draw face box even without prediction
            cv2.rectangle(frame, (face_bbox[0], face_bbox[1]),
                         (face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]),
                         (255, 255, 255), 2)
            
            # Show real-time metrics for debugging
            ear_avg = (features[0] + features[1]) / 2.0
            mar = features[2]
            metrics_text = f"EAR: {ear_avg:.3f} | MAR: {mar:.3f}"
            cv2.putText(frame, metrics_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show buffering status if not full
            if len(self.face_buffer) < 16:
                buffer_status = f"Buffering: {len(self.face_buffer)}/16"
                cv2.putText(frame, buffer_status, (10, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self, fps_target=30):
        """
        Run real-time detection loop
        
        Args:
            fps_target: Target frames per second
        """
        # Open video capture
        if isinstance(self.camera_id, int):
            cap = cv2.VideoCapture(self.camera_id)
            print(f"Opened camera {self.camera_id}")
        else:
            cap = cv2.VideoCapture(self.camera_id)
            print(f"Opened video file: {self.camera_id}")
        
        if not cap.isOpened():
            print(f"Error: Could not open camera/video {self.camera_id}")
            return
        
        # Set camera properties for better performance
        if isinstance(self.camera_id, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, fps_target)
        
        # Update feature extractor FPS
        self.feature_extractor.set_fps(fps_target)
        
        print("\n" + "="*70)
        print("Starting drowsiness detection...")
        print("Controls: 'q' = Quit | 's' = Statistics | 'c' = Recalibrate")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame (end of video or camera error)")
                    break
                
                # Process frame
                timestamp = time.time() - start_time
                processed_frame, prediction = self.process_frame(frame, timestamp)
                
                # Calculate FPS
                loop_time = time.time() - loop_start
                current_fps = 1.0 / loop_time if loop_time > 0 else 0
                self.fps_list.append(current_fps)
                
                # Keep only last 30 FPS values
                if len(self.fps_list) > 30:
                    self.fps_list.pop(0)
                
                avg_fps = np.mean(self.fps_list)
                
                # Draw FPS
                processed_frame = draw_fps(processed_frame, avg_fps, position=(10, 30))
                
                # Display frame
                cv2.imshow('Driver Drowsiness Detection', processed_frame)
                
                self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    stats = self.alert_system.get_statistics()
                    print("\n" + "="*50)
                    print("Session Statistics:")
                    print(f"  Total frames processed: {self.frame_count}")
                    print(f"  Average FPS: {avg_fps:.1f}")
                    print(f"  Alert events: {stats['alert_count']}")
                    print(f"  Warning events: {stats['warning_count']}")
                    print(f"  Drowsy events: {stats['drowsy_count']}")
                    print("="*50 + "\n")
                elif key == ord('c'):
                    print("\n[Manual Trigger] Starting on-demand personal baseline calibration...")
                    self.feature_extractor.start_calibration(duration=5.0)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'detector'):
                self.detector.close()
            self.alert_system.cleanup()
            
            if self.video_recorder:
                self.video_recorder.cleanup()
            
            # Print final statistics
            print("\n" + "="*70)
            print("Session Summary")
            print("="*70)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Average FPS: {np.mean(self.fps_list):.1f}")
            
            stats = self.alert_system.get_statistics()
            print(f"\nAlert Statistics:")
            print(f"  Alert events: {stats['alert_count']}")
            print(f"  Warning events: {stats['warning_count']}")
            print(f"  Drowsy events: {stats['drowsy_count']}")
            
            print(f"\nLogs saved to: {self.output_dirs['logs']}")
            if self.enable_recording:
                print(f"Recordings saved to: {self.output_dirs['recordings']}")
            print("="*70 + "\n")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Real-Time Driver Drowsiness Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with webcam
  python main.py
  
  # Run with specific camera
  python main.py --source 1
  
  # Run with video file
  python main.py --source video.mp4
  
  # Run with custom model
  python main.py --model models/my_model.h5
  
  # Run without audio alerts
  python main.py --no-audio
  
  # Run in debug mode
  python main.py --debug
        """
    )
    
    parser.add_argument('--source', type=str, default='0',
                       help='Camera ID or video file path (default: 0)')
    parser.add_argument('--model', type=str, default='models/drowsiness_detector.h5',
                       help='Path to trained model (default: models/drowsiness_detector.h5)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for logs and recordings (default: output)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio alerts')
    parser.add_argument('--no-recording', action='store_true',
                       help='Disable video recording')
    parser.add_argument('--no-calibration', action='store_true',
                       help='Skip initial 5-second personal baseline calibration')
    parser.add_argument('--no-clahe', action='store_true',
                       help='Disable adaptive low-light contrast enhancement')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information and landmarks')
    
    args = parser.parse_args()
    
    # Parse source (convert to int if it's a number)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\n⚠ Warning: Model file not found at {args.model}")
        print("The system will run in feature extraction mode only.")
        print("To train a model first, run: python train.py\n")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    # Initialize and run system
    system = DrowsinessDetectionSystem(
        model_path=args.model,
        camera_id=source,
        enable_audio=not args.no_audio,
        enable_recording=not args.no_recording,
        output_dir=args.output_dir,
        debug=args.debug,
        enable_calibration=not args.no_calibration,
        enable_low_light=not args.no_clahe
    )
    
    system.run(fps_target=args.fps)


if __name__ == "__main__":
    main()
