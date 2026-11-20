"""
Utility Functions for Driver Drowsiness Detection System
Handles preprocessing, sequence buffering, video recording, and helper functions
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os


def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two 2D/3D points
    
    Args:
        p1: First point (x, y) or (x, y, z)
        p2: Second point (x, y) or (x, y, z)
    
    Returns:
        float: Euclidean distance
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def preprocess_face_crop(face_img, target_size=(48, 48), use_clahe=True):
    """
    Preprocess face crop for CNN input
    - Resize to target size
    - Convert to grayscale
    - Optionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
      to enhance nighttime / low-light driving visibility
    - Normalize to [0, 1]
    
    Args:
        face_img: Face region image (BGR)
        target_size: Target dimensions (width, height)
        use_clahe: Enable low-light adaptive contrast enhancement
    
    Returns:
        np.array: Preprocessed face image (target_size, 1 channel)
    """
    # Convert to grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Apply CLAHE to handle low-light or uneven cabin illumination
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add channel dimension
    preprocessed = np.expand_dims(normalized, axis=-1)
    
    return preprocessed


def enhance_low_light_frame(frame, threshold_luma=75):
    """
    Dynamically enhance low-light cabin video frames using CLAHE on L-channel
    in LAB color space (preserves natural colors while brightening shadows).
    
    Args:
        frame: BGR video frame
        threshold_luma: Luminance threshold below which enhancement triggers
        
    Returns:
        np.array: Enhanced BGR video frame
    """
    # Fast check of average brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    
    if mean_brightness < threshold_luma:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Merge channels back and convert to BGR
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return frame


def normalize_landmarks(landmarks, frame_width, frame_height):
    """
    Normalize landmark coordinates to [0, 1] range relative to frame size
    
    Args:
        landmarks: MediaPipe landmark list
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
    Returns:
        np.array: Normalized landmarks (N, 3) where N is number of landmarks
    """
    normalized = []
    for lm in landmarks:
        normalized.append([
            lm.x,  # Already normalized by MediaPipe
            lm.y,
            lm.z
        ])
    return np.array(normalized)


def extract_face_roi(frame, landmarks, frame_width, frame_height, padding=0.2):
    """
    Extract face region of interest with padding
    
    Args:
        frame: Input frame (BGR)
        landmarks: MediaPipe normalized landmarks
        frame_width: Frame width
        frame_height: Frame height
        padding: Padding ratio around face (default 20%)
    
    Returns:
        np.array: Face ROI image, or None if extraction fails
    """
    # Get bounding box from landmarks
    x_coords = [lm.x * frame_width for lm in landmarks]
    y_coords = [lm.y * frame_height for lm in landmarks]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, int(x_min - width * padding))
    x_max = min(frame_width, int(x_max + width * padding))
    y_min = max(0, int(y_min - height * padding))
    y_max = min(frame_height, int(y_max + height * padding))
    
    # Extract ROI
    face_roi = frame[y_min:y_max, x_min:x_max]
    
    if face_roi.size == 0:
        return None
    
    return face_roi


class SequenceBuffer:
    """
    Manages sliding window buffer of frames/features for temporal analysis
    Uses fixed-size deque for efficient append/pop operations
    """
    
    def __init__(self, max_length=16):
        """
        Initialize sequence buffer
        
        Args:
            max_length: Maximum number of frames to store (default 16)
        """
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)
    
    def add(self, item):
        """
        Add item to buffer (automatically removes oldest if full)
        
        Args:
            item: Frame, feature vector, or any data to buffer
        """
        self.buffer.append(item)
    
    def get_sequence(self):
        """
        Get current sequence as numpy array
        
        Returns:
            np.array: Stacked sequence, or None if buffer not full
        """
        if len(self.buffer) < self.max_length:
            return None
        
        return np.array(list(self.buffer))
    
    def is_full(self):
        """
        Check if buffer has reached maximum length
        
        Returns:
            bool: True if buffer is full
        """
        return len(self.buffer) == self.max_length
    
    def clear(self):
        """Clear all items from buffer"""
        self.buffer.clear()
    
    def __len__(self):
        """Get current buffer size"""
        return len(self.buffer)


class VideoRecorder:
    """
    Records video clips of drowsy events for later review
    Handles pre-event buffering and automatic file saving
    """
    
    def __init__(self, output_dir='recordings', fps=30, pre_event_seconds=5):
        """
        Initialize video recorder
        
        Args:
            output_dir: Directory to save recordings
            fps: Frames per second
            pre_event_seconds: Seconds to record before event trigger
        """
        self.output_dir = output_dir
        self.fps = fps
        self.pre_event_frames = int(fps * pre_event_seconds)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Frame buffer for pre-event recording
        self.frame_buffer = deque(maxlen=self.pre_event_frames)
        
        # Current recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_frames = 0
        self.max_recording_frames = fps * 15  # 15 seconds max
    
    def add_frame(self, frame):
        """
        Add frame to pre-event buffer
        
        Args:
            frame: Video frame (BGR)
        """
        self.frame_buffer.append(frame.copy())
    
    def start_recording(self, frame):
        """
        Start recording drowsy event
        Writes buffered pre-event frames first
        
        Args:
            frame: Current frame that triggered recording
        """
        if self.is_recording:
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"drowsy_event_{timestamp}.avi")
        
        # Initialize video writer
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        
        # Write buffered pre-event frames
        for buffered_frame in self.frame_buffer:
            self.video_writer.write(buffered_frame)
        
        # Write current frame
        self.video_writer.write(frame)
        
        self.is_recording = True
        self.recording_frames = 1
        
        print(f"[VideoRecorder] Started recording: {filename}")
    
    def record_frame(self, frame):
        """
        Record frame if currently recording
        
        Args:
            frame: Video frame (BGR)
        """
        if not self.is_recording:
            return
        
        self.video_writer.write(frame)
        self.recording_frames += 1
        
        # Auto-stop if max duration reached
        if self.recording_frames >= self.max_recording_frames:
            self.stop_recording()
    
    def stop_recording(self):
        """Stop current recording and save file"""
        if not self.is_recording:
            return
        
        self.video_writer.release()
        self.is_recording = False
        self.recording_frames = 0
        
        print(f"[VideoRecorder] Stopped recording")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.is_recording:
            self.stop_recording()


def draw_fps(frame, fps, position=(10, 30)):
    """
    Draw FPS counter on frame
    
    Args:
        frame: Video frame
        fps: Current FPS value
        position: Text position (x, y)
    
    Returns:
        np.array: Frame with FPS overlay
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    return frame


def draw_landmarks(frame, landmarks, frame_width, frame_height, connections=None):
    """
    Draw facial landmarks on frame for visualization
    
    Args:
        frame: Video frame
        landmarks: MediaPipe landmark list
        frame_width: Frame width
        frame_height: Frame height
        connections: Optional landmark connections to draw
    
    Returns:
        np.array: Frame with landmarks drawn
    """
    for lm in landmarks:
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    return frame


def create_output_dirs(base_dir='output'):
    """
    Create necessary output directories for logs and recordings
    
    Args:
        base_dir: Base output directory
    
    Returns:
        dict: Dictionary with paths to created directories
    """
    dirs = {
        'base': base_dir,
        'logs': os.path.join(base_dir, 'logs'),
        'recordings': os.path.join(base_dir, 'recordings'),
        'models': os.path.join(base_dir, 'models')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs
