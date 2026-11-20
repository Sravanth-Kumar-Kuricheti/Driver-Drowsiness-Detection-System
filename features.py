"""
Feature Extraction Module for Driver Drowsiness Detection
Implements EAR, MAR, head pose estimation, and 16-feature vector extraction
Uses MediaPipe FaceMesh landmarks
"""

import cv2
import numpy as np
from collections import deque
from utils import calculate_distance


# MediaPipe FaceMesh landmark indices (468-point model)
# Eye landmarks (using subset of 468 landmarks)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 6 points around left eye
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 6 points around right eye

# Mouth landmarks (Left corner, Right corner, Top lip, Bottom lip)
MOUTH_INDICES = [78, 308, 13, 14]

# Head pose reference points (nose, chin, left eye, right eye, left mouth, right mouth)
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]


class FeatureExtractor:
    """
    Extracts drowsiness-related features from facial landmarks
    Computes EAR, MAR, head pose, and temporal features
    """
    
    def __init__(self):
        """Initialize feature extractor with thresholds and state tracking"""
        
        # Base Thresholds (adaptable via calibration)
        self.DEFAULT_EAR_THRESHOLD = 0.20
        self.EAR_THRESHOLD = 0.20
        self.DEFAULT_MAR_THRESHOLD = 0.60
        self.MAR_THRESHOLD = 0.60
        self.BLINK_DURATION_THRESHOLD = 0.45  # seconds
        self.YAWN_DURATION_THRESHOLD = 1.8   # seconds
        
        # Personal Calibration state
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_duration = 5.0
        self.calibration_ear_samples = []
        self.calibration_mar_samples = []
        self.is_calibrated = False
        self.baseline_ear = None
        self.baseline_mar = None
        
        # PERCLOS (Percentage of Eye Closure over rolling 30-sec window)
        self.perclos_window_seconds = 30.0
        self.perclos_history = deque()  # stores (timestamp, is_closed)
        self.current_perclos = 0.0
        
        # State tracking
        self.eye_closed_start = None
        self.eye_closure_duration = 0.0
        self.mouth_open_start = None
        self.mouth_open_duration = 0.0
        self.previous_head_pose = None
        self.frame_count = 0
        self.fps = 30  # Default, will be updated
        
        # 3D model points for head pose estimation (generic face model in cm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -6.3, -3.9),         # Chin
            (-4.3, 3.3, -3.1),         # Left eye corner
            (4.3, 3.3, -3.1),          # Right eye corner
            (-2.9, -2.5, -2.0),        # Left mouth corner
            (2.9, -2.5, -2.0)          # Right mouth corner
        ], dtype=np.float64)
    
    def set_fps(self, fps):
        """
        Set frame rate for time-based calculations
        
        Args:
            fps: Frames per second
        """
        self.fps = fps
        self.perclos_window_seconds = 30.0

    def start_calibration(self, duration=5.0):
        """
        Begin personal calibration session to adapt EAR and MAR thresholds
        to driver's specific facial morphology.
        
        Args:
            duration: Calibration duration in seconds (default 5.0)
        """
        self.is_calibrating = True
        self.calibration_start_time = None
        self.calibration_duration = duration
        self.calibration_ear_samples = []
        self.calibration_mar_samples = []
        print("\n[Calibration] Started 5-second personal baseline calibration...")
        print("[Calibration] Look straight ahead at the road naturally.")

    def update_calibration(self, ear, mar, timestamp):
        """
        Update calibration with current frame's metrics
        
        Returns:
            tuple: (is_calibrating, progress_ratio 0.0-1.0, just_completed)
        """
        if not self.is_calibrating:
            return False, 1.0, False
            
        if self.calibration_start_time is None:
            self.calibration_start_time = timestamp
            
        elapsed = timestamp - self.calibration_start_time
        progress = min(1.0, elapsed / self.calibration_duration)
        
        # Collect samples when values are reasonable (eyes open, head mostly forward)
        if 0.15 <= ear <= 0.45:
            self.calibration_ear_samples.append(ear)
        if 0.1 <= mar <= 0.8:
            self.calibration_mar_samples.append(mar)
            
        # Check if calibration completed
        if elapsed >= self.calibration_duration and len(self.calibration_ear_samples) >= 15:
            # Baseline is driver's resting EAR/MAR (using 50th percentile / median)
            self.baseline_ear = float(np.median(self.calibration_ear_samples))
            self.baseline_mar = float(np.median(self.calibration_mar_samples))
            
            # Dynamic personalized threshold: 72% of normal resting EAR
            self.EAR_THRESHOLD = round(max(0.16, min(0.26, self.baseline_ear * 0.72)), 3)
            # Dynamic yawning threshold: 1.6x of resting mouth opening
            self.MAR_THRESHOLD = round(max(0.50, min(0.75, self.baseline_mar * 1.65)), 3)
            
            self.is_calibrating = False
            self.is_calibrated = True
            print(f"[Calibration] ✓ Calibration complete!")
            print(f"[Calibration] Baseline EAR: {self.baseline_ear:.3f} -> Personalized Threshold: {self.EAR_THRESHOLD:.3f}")
            print(f"[Calibration] Baseline MAR: {self.baseline_mar:.3f} -> Personalized Threshold: {self.MAR_THRESHOLD:.3f}\n")
            return False, 1.0, True
            
        return True, progress, False
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: List of 6 eye landmark points [(x,y), ...]
        
        Returns:
            float: Eye aspect ratio value
        """
        # Vertical distances
        vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
        vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3])
        
        # Avoid division by zero
        if horizontal < 1e-6:
            return 0.0
        
        # Calculate EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR)
        MAR = (vertical distance) / (horizontal distance)
        
        Args:
            mouth_landmarks: List of mouth landmark points
        
        Returns:
            float: Mouth aspect ratio value
        """
        # Vertical distance (Top to Bottom center)
        vertical = calculate_distance(mouth_landmarks[2], mouth_landmarks[3])
        
        # Horizontal distance (Left to Right mouth corners)
        horizontal = calculate_distance(mouth_landmarks[0], mouth_landmarks[1])
        
        # Avoid division by zero
        if horizontal < 1e-6:
            return 0.0
        
        # Calculate MAR
        mar = vertical / horizontal
        
        return mar
    
    def estimate_head_pose(self, landmarks, frame_width, frame_height):
        """
        Estimate head pose (yaw, pitch, roll) using solvePnP
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        
        Returns:
            tuple: (yaw, pitch, roll) in degrees, or (0, 0, 0) if estimation fails
        """
        # Extract 2D image points for head pose indices
        image_points = []
        for idx in HEAD_POSE_INDICES:
            landmark = landmarks[idx]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            image_points.append([x, y])
        
        image_points = np.array(image_points, dtype=np.float64)
        
        # Camera matrix (assuming centered principal point)
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles from rotation matrix
        # Using the convention: yaw (Y), pitch (X), roll (Z)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        
        return yaw, pitch, roll
    
    def extract_features(self, landmarks, frame_width, frame_height, timestamp):
        """
        Extract all 16 features from facial landmarks
        
        Args:
            landmarks: MediaPipe face mesh landmarks (468 points)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            timestamp: Current frame timestamp
        
        Returns:
            np.array: 16-dimensional feature vector
        """
        # Convert landmarks to pixel coordinates
        landmark_coords = []
        for lm in landmarks.landmark:
            landmark_coords.append([lm.x * frame_width, lm.y * frame_height])
        landmark_coords = np.array(landmark_coords)
        
        # --- 1-2: EAR for left and right eyes ---
        left_eye_points = [landmark_coords[i] for i in LEFT_EYE_INDICES]
        right_eye_points = [landmark_coords[i] for i in RIGHT_EYE_INDICES]
        
        ear_left = self.calculate_ear(left_eye_points)
        ear_right = self.calculate_ear(right_eye_points)
        ear_avg = (ear_left + ear_right) / 2.0
        
        # --- 3: MAR (Mouth Aspect Ratio) ---
        mouth_points = [landmark_coords[i] for i in MOUTH_INDICES]
        mar = self.calculate_mar(mouth_points)
        
        # --- 4-5: Blink and Yawn status ---
        # Track eye closure duration
        if ear_avg < self.EAR_THRESHOLD:
            if self.eye_closed_start is None:
                self.eye_closed_start = timestamp
            self.eye_closure_duration = timestamp - self.eye_closed_start
            blink_status = 1
        else:
            self.eye_closed_start = None
            self.eye_closure_duration = 0.0
            blink_status = 0
        
        # Track yawn duration
        if mar > self.MAR_THRESHOLD:
            if self.mouth_open_start is None:
                self.mouth_open_start = timestamp
            self.mouth_open_duration = timestamp - self.mouth_open_start
            yawn_status = 1
        else:
            self.mouth_open_start = None
            self.mouth_open_duration = 0.0
            yawn_status = 0
        
        # --- 6-8: Head pose (yaw, pitch, roll) ---
        yaw, pitch, roll = self.estimate_head_pose(landmarks.landmark, frame_width, frame_height)
        
        # --- 9-10: Eye closure and mouth open durations ---
        # Already calculated above
        
        # --- 11-13: Head velocity (change in head pose) ---
        if self.previous_head_pose is not None:
            head_velocity_x = yaw - self.previous_head_pose[0]
            head_velocity_y = pitch - self.previous_head_pose[1]
            head_velocity_z = roll - self.previous_head_pose[2]
        else:
            head_velocity_x = 0.0
            head_velocity_y = 0.0
            head_velocity_z = 0.0
        
        self.previous_head_pose = [yaw, pitch, roll]
        
        # --- 14-15: Gaze direction (approximated from head pose) ---
        # Simplified: use yaw and pitch as gaze direction
        gaze_x = np.sin(np.radians(yaw))
        gaze_y = np.sin(np.radians(pitch))
        
        # --- 16: Drowsiness score (heuristic combination) ---
        # Weighted combination of features
        drowsiness_score = 0.0
        
        # Low EAR contributes to drowsiness
        if ear_avg < self.EAR_THRESHOLD:
            drowsiness_score += 0.3
        
        # Long eye closure contributes
        if self.eye_closure_duration > self.BLINK_DURATION_THRESHOLD:
            drowsiness_score += 0.2
        
        # Yawning contributes
        if yawn_status == 1:
            drowsiness_score += 0.2
        
        # Head drooping (pitch forward) contributes
        if pitch > 25:
            drowsiness_score += 0.15
        
        # Head turned away contributes
        if abs(yaw) > 35:
            drowsiness_score += 0.1
        
        # Head tilted contributes
        if abs(roll) > 20:
            drowsiness_score += 0.05
        
        # Clip to [0, 1]
        drowsiness_score = min(1.0, drowsiness_score)
        
        # Assemble feature vector
        features = np.array([
            ear_left,                    # 1
            ear_right,                   # 2
            mar,                         # 3
            blink_status,                # 4
            yawn_status,                 # 5
            yaw,                         # 6
            pitch,                       # 7
            roll,                        # 8
            self.eye_closure_duration,   # 9
            self.mouth_open_duration,    # 10
            head_velocity_x,             # 11
            head_velocity_y,             # 12
            head_velocity_z,             # 13
            gaze_x,                      # 14
            gaze_y,                      # 15
            drowsiness_score             # 16
        ], dtype=np.float32)
        
    def estimate_head_pose_task(self, landmarks, frame_width, frame_height):
        """
        Estimate head pose using landmarks from Tasks API
        """
        image_points = []
        for idx in HEAD_POSE_INDICES:
            landmark = landmarks[idx]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            image_points.append([x, y])
        
        image_points = np.array(image_points, dtype=np.float64)
        
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
            pitch = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        else:
            yaw = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
            pitch = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = 0
            
        return yaw, pitch, roll

    def extract_features_task(self, landmarks, frame_width, frame_height, timestamp):
        """
        Extract features from landmarks provided by Tasks API (List[NormalizedLandmark])
        """
        # Convert landmarks to pixel coordinates
        landmark_coords = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in landmarks])
        
        # 1-2: EAR
        left_eye_points = landmark_coords[LEFT_EYE_INDICES]
        right_eye_points = landmark_coords[RIGHT_EYE_INDICES]
        ear_left = self.calculate_ear(left_eye_points)
        ear_right = self.calculate_ear(right_eye_points)
        ear_avg = (ear_left + ear_right) / 2.0
        
        # 3: MAR
        mouth_points = landmark_coords[MOUTH_INDICES]
        mar = self.calculate_mar(mouth_points)
        
        # 4-5: Blink and Yawn status
        is_eye_closed = ear_avg < self.EAR_THRESHOLD
        if is_eye_closed:
            if self.eye_closed_start is None: self.eye_closed_start = timestamp
            self.eye_closure_duration = timestamp - self.eye_closed_start
            blink_status = 1
        else:
            self.eye_closed_start = None
            self.eye_closure_duration = 0.0
            blink_status = 0
            
        # Update PERCLOS (Percentage of Eye Closure over rolling window)
        self.perclos_history.append((timestamp, 1 if is_eye_closed else 0))
        # Remove items older than window
        cutoff_time = timestamp - self.perclos_window_seconds
        while self.perclos_history and self.perclos_history[0][0] < cutoff_time:
            self.perclos_history.popleft()
            
        if self.perclos_history:
            closed_count = sum(flag for _, flag in self.perclos_history)
            self.current_perclos = closed_count / len(self.perclos_history)
        else:
            self.current_perclos = 0.0
            
        if mar > self.MAR_THRESHOLD:
            if self.mouth_open_start is None: self.mouth_open_start = timestamp
            self.mouth_open_duration = timestamp - self.mouth_open_start
            yawn_status = 1
        else:
            self.mouth_open_start = None
            self.mouth_open_duration = 0.0
            yawn_status = 0
            
        # 6-8: Head pose
        yaw, pitch, roll = self.estimate_head_pose_task(landmarks, frame_width, frame_height)
        
        # 11-13: Head velocity
        if self.previous_head_pose is not None:
            head_velocity_x = yaw - self.previous_head_pose[0]
            head_velocity_y = pitch - self.previous_head_pose[1]
            head_velocity_z = roll - self.previous_head_pose[2]
        else:
            head_velocity_x = head_velocity_y = head_velocity_z = 0.0
        self.previous_head_pose = [yaw, pitch, roll]
        
        # 14-15: Gaze
        gaze_x = np.sin(np.radians(yaw))
        gaze_y = np.sin(np.radians(pitch))
        
        # 16: Multi-Modal Biometric Drowsiness Score
        drowsiness_score = 0.0
        # Instantaneous Eye Closure
        if ear_avg < self.EAR_THRESHOLD: 
            drowsiness_score += 0.25
        # Prolonged Blink / Microsleep
        if self.eye_closure_duration > self.BLINK_DURATION_THRESHOLD: 
            drowsiness_score += min(0.35, 0.20 + (self.eye_closure_duration - self.BLINK_DURATION_THRESHOLD) * 0.3)
        # PERCLOS factor (Golden standard: >15% is warning, >25% is severe drowsiness)
        if self.current_perclos > 0.12:
            drowsiness_score += min(0.30, (self.current_perclos - 0.12) * 2.0)
        # Yawning
        if yawn_status == 1: 
            drowsiness_score += 0.20
        # Head Drooping / Nodding off
        if pitch > 22: 
            drowsiness_score += 0.15
        # Distraction / Head turned away
        if abs(yaw) > 32: 
            drowsiness_score += 0.10
        if abs(roll) > 18: 
            drowsiness_score += 0.05
            
        drowsiness_score = min(1.0, max(0.0, drowsiness_score))
        
        features = np.array([
            ear_left, ear_right, mar, blink_status, yawn_status,
            yaw, pitch, roll, self.eye_closure_duration, self.mouth_open_duration,
            head_velocity_x, head_velocity_y, head_velocity_z,
            gaze_x, gaze_y, drowsiness_score
        ], dtype=np.float32)
        
        self.frame_count += 1
        return features

    def get_debug_info(self):
        """
        Get current feature values for debugging/visualization
        
        Returns:
            dict: Dictionary with current feature states
        """
        return {
            'eye_closure_duration': self.eye_closure_duration,
            'mouth_open_duration': self.mouth_open_duration,
            'head_pose': self.previous_head_pose if self.previous_head_pose else [0, 0, 0],
            'frame_count': self.frame_count,
            'perclos': self.current_perclos,
            'is_calibrated': self.is_calibrated,
            'baseline_ear': self.baseline_ear,
            'baseline_mar': self.baseline_mar,
            'ear_threshold': self.EAR_THRESHOLD,
            'mar_threshold': self.MAR_THRESHOLD
        }
