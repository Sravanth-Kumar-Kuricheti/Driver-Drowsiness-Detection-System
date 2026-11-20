"""
Alert System for Driver Drowsiness Detection
Implements visual and audio alerts with escalation logic
"""

import cv2
import numpy as np
import pygame
import pyttsx3
import threading
import time
import csv
import os
from datetime import datetime
from collections import deque

try:
    import winsound
except ImportError:
    winsound = None


class AlertSystem:
    """
    Manages multi-modal alerts (visual + audio) with escalation
    Logs all drowsiness events to CSV
    """
    
    # State definitions
    STATE_ALERT = 0
    STATE_WARNING = 1
    STATE_DROWSY = 2
    
    # Color codes for visualization (BGR format)
    COLOR_ALERT = (0, 255, 0)      # Green
    COLOR_WARNING = (0, 255, 255)  # Yellow
    COLOR_DROWSY = (0, 0, 255)     # Red
    
    def __init__(self, log_dir='logs', enable_audio=True, enable_voice=True):
        """
        Initialize alert system
        
        Args:
            log_dir: Directory for CSV logs
            enable_audio: Enable audio alerts
            enable_voice: Enable text-to-speech alerts
        """
        self.log_dir = log_dir
        self.enable_audio = enable_audio
        self.enable_voice = enable_voice
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV logger
        self.log_file = os.path.join(
            log_dir, 
            f"drowsiness_log_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        self._init_csv_log()
        
        # Initialize audio system with Windows fallback
        self.use_winsound = False
        self.beep_sound = None
        self.alarm_sound = None
        
        if enable_audio:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.beep_sound = self._generate_beep(frequency=800, duration=0.3)
                self.alarm_sound = self._generate_beep(frequency=1200, duration=1.0)
            except Exception as e:
                print(f"[AlertSystem] Note: Pygame audio unavailable ({e}). Using native Windows sound.")
                self.use_winsound = (winsound is not None)
        
        # Initialize text-to-speech
        if enable_voice:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)
            except:
                print("[AlertSystem] Warning: Text-to-speech not available")
                self.enable_voice = False
        
        # State tracking
        self.current_state = self.STATE_ALERT
        self.previous_state = self.STATE_ALERT
        self.state_change_time = time.time()
        self.last_alert_time = 0
        self.alert_interval = 5.0  # Escalate every 5 seconds
        
        # Event history for analysis
        self.event_history = deque(maxlen=1000)
        
        # Telemetry history for real-time live HUD strip chart
        self.telemetry_history = deque(maxlen=80)
    
    def _init_csv_log(self):
        """Initialize CSV log file with headers"""
        file_exists = os.path.exists(self.log_file)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp',
                    'state',
                    'state_name',
                    'confidence',
                    'ear_left',
                    'ear_right',
                    'mar',
                    'yaw',
                    'pitch',
                    'roll',
                    'eye_closure_duration',
                    'mouth_open_duration',
                    'drowsiness_score'
                ])
    
    def _generate_beep(self, frequency=800, duration=0.5):
        """
        Generate beep sound using pygame
        
        Args:
            frequency: Beep frequency in Hz
            duration: Beep duration in seconds
        
        Returns:
            pygame.Sound: Generated beep sound
        """
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        
        # Generate sine wave
        t = np.linspace(0, duration, n_samples, False)
        wave = np.sin(frequency * 2 * np.pi * t)
        
        # Apply fade in/out to avoid clicks
        fade_len = int(sample_rate * 0.01)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        wave[:fade_len] *= fade_in
        wave[-fade_len:] *= fade_out
        
        # Convert to 16-bit audio
        audio = (wave * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_audio = np.column_stack((audio, audio))
        
        # Create pygame Sound
        sound = pygame.sndarray.make_sound(stereo_audio)
        
        return sound
    
    def _play_audio_async(self, sound):
        """
        Play audio in background thread
        
        Args:
            sound: pygame.Sound object
        """
        if sound is None:
            return
        def play():
            try:
                sound.play()
            except Exception:
                pass
        
        threading.Thread(target=play, daemon=True).start()

    def _play_winsound_async(self, frequency=800, duration_ms=300):
        """
        Play native Windows beep in background thread
        """
        if winsound is None:
            return
        def play():
            try:
                winsound.Beep(int(frequency), int(duration_ms))
            except Exception:
                pass
        threading.Thread(target=play, daemon=True).start()
    
    def _speak_async(self, text):
        """
        Speak text asynchronously using TTS
        
        Args:
            text: Text to speak
        """
        if not self.enable_voice:
            return
        
        def speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass
        
        threading.Thread(target=speak, daemon=True).start()
    
    def update_state(self, prediction_result, features=None):
        """
        Update alert state based on prediction
        
        Args:
            prediction_result: Dictionary from DrowsinessPredictor
            features: Optional feature vector for logging
        
        Returns:
            bool: True if state changed
        """
        new_state = prediction_result['state']
        current_time = time.time()
        
        # Check if state changed
        state_changed = (new_state != self.current_state)
        
        if state_changed:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_change_time = current_time
            
            # Log state change
            self._log_event(prediction_result, features)
            
            # Trigger immediate alert on state change
            self._trigger_alert(new_state, is_escalation=False)
        
        # Check for escalation (same dangerous state for too long)
        elif new_state in [self.STATE_WARNING, self.STATE_DROWSY]:
            time_since_last_alert = current_time - self.last_alert_time
            
            if time_since_last_alert >= self.alert_interval:
                self._trigger_alert(new_state, is_escalation=True)
        
        return state_changed
    
    def _trigger_alert(self, state, is_escalation=False):
        """
        Trigger appropriate alert based on state
        
        Args:
            state: Current drowsiness state
            is_escalation: Whether this is an escalation alert
        """
        self.last_alert_time = time.time()
        
        if state == self.STATE_ALERT:
            # No alerts for alert state
            pass
        
        elif state == self.STATE_WARNING:
            # Warning: Soft beep (via pygame or native winsound fallback)
            if self.enable_audio:
                if self.use_winsound:
                    self._play_winsound_async(850, 220)
                else:
                    self._play_audio_async(self.beep_sound)
            
            if is_escalation:
                self._speak_async("Stay alert")
        
        elif state == self.STATE_DROWSY:
            # Drowsy: Loud alarm + voice warning (via pygame or native winsound fallback)
            if self.enable_audio:
                if self.use_winsound:
                    self._play_winsound_async(1200, 500)
                else:
                    self._play_audio_async(self.alarm_sound)
            
            if is_escalation:
                self._speak_async("PULL OVER NOW!")
            else:
                self._speak_async("Eyes open! Wake up!")
    
    def _log_event(self, prediction_result, features):
        """
        Log drowsiness event to CSV
        
        Args:
            prediction_result: Prediction dictionary
            features: Feature vector (16 dimensions)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Extract feature values if available
        if features is not None and len(features) >= 16:
            ear_left = features[0]
            ear_right = features[1]
            mar = features[2]
            yaw = features[5]
            pitch = features[6]
            roll = features[7]
            eye_closure = features[8]
            mouth_open = features[9]
            drowsy_score = features[15]
        else:
            ear_left = ear_right = mar = yaw = pitch = roll = 0.0
            eye_closure = mouth_open = drowsy_score = 0.0
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                prediction_result['state'],
                prediction_result['state_name'],
                prediction_result['confidence'],
                ear_left,
                ear_right,
                mar,
                yaw,
                pitch,
                roll,
                eye_closure,
                mouth_open,
                drowsy_score
            ])
        
        # Add to event history
        self.event_history.append({
            'timestamp': timestamp,
            'state': prediction_result['state']
        })
    
    def draw_visual_alerts(self, frame, prediction_result, features=None, face_bbox=None,
                           calibration_info=None, perclos=0.0):
        """
        Draw visual alerts, status bar, and real-time live telemetry strip chart
        
        Args:
            frame: Video frame (BGR)
            prediction_result: Prediction dictionary
            features: Optional feature vector for display
            face_bbox: Optional face bounding box (x, y, w, h)
            calibration_info: Optional dict with calibration status and baseline
            perclos: Current PERCLOS metric ratio (0.0 to 1.0)
        
        Returns:
            np.array: Frame with visual alerts and telemetry drawn
        """
        state = prediction_result['state']
        state_name = prediction_result['state_name']
        confidence = prediction_result['confidence']
        h_frame, w_frame = frame.shape[:2]
        
        # Get color based on state
        if state == self.STATE_ALERT:
            color = self.COLOR_ALERT
        elif state == self.STATE_WARNING:
            color = self.COLOR_WARNING
        else:
            color = self.COLOR_DROWSY
        
        # Draw face bounding box if provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Draw status bar at top
        overlay = frame.copy()
        bar_height = 80
        cv2.rectangle(overlay, (0, 0), (w_frame, bar_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw state text
        state_text = f"STATE: {state_name}"
        cv2.putText(frame, state_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw confidence and PERCLOS
        conf_text = f"Confidence: {confidence:.2f} | PERCLOS: {perclos * 100:.1f}%"
        cv2.putText(frame, conf_text, (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        # Draw Calibration Badge in top-right if calibrated
        if calibration_info and calibration_info.get('is_calibrated'):
            b_ear = calibration_info.get('baseline_ear', 0.28)
            t_ear = calibration_info.get('ear_threshold', 0.20)
            cal_badge = f"CALIBRATED (Base EAR:{b_ear:.2f}->Thresh:{t_ear:.2f})"
            cv2.putText(frame, cal_badge, (w_frame - 390, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
        
        # Draw feature metrics on left side
        if features is not None and len(features) >= 16:
            ear_val = (features[0] + features[1]) / 2.0
            mar_val = features[2]
            drowsy_val = features[15]
            
            # Record telemetry point
            self.telemetry_history.append((ear_val, mar_val, drowsy_val))
            
            y_offset = 110
            metrics = [
                f"EAR: {ear_val:.3f}",
                f"MAR: {mar_val:.3f}",
                f"Head: Y={features[5]:.1f} P={features[6]:.1f} R={features[7]:.1f}",
                f"Eye Closed: {features[8]:.2f}s",
                f"Drowsy Score: {drowsy_val:.3f}"
            ]
            
            for i, metric in enumerate(metrics):
                cv2.putText(frame, metric, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw Real-Time Telemetry HUD Strip Chart in bottom-right
        if len(self.telemetry_history) >= 2:
            card_w, card_h = 240, 110
            card_x = w_frame - card_w - 10
            card_y = h_frame - card_h - 10
            
            # Card background
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
            cv2.rectangle(hud_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)
            frame = cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0)
            
            # Card Title & Legend
            cv2.putText(frame, "TELEMETRY (EAR:Green | MAR:Orange)", (card_x + 8, card_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            
            # Plot bounds
            plot_x = card_x + 8
            plot_y = card_y + 24
            plot_w = card_w - 16
            plot_h = card_h - 32
            
            # Grid line at EAR threshold ~ 0.20
            thresh_val = calibration_info.get('ear_threshold', 0.20) if calibration_info else 0.20
            thresh_y = int(plot_y + plot_h - (thresh_val / 0.5) * plot_h)
            if plot_y <= thresh_y <= plot_y + plot_h:
                cv2.line(frame, (plot_x, thresh_y), (plot_x + plot_w, thresh_y), (0, 100, 255), 1)
            
            # Draw sparklines for EAR and MAR
            n_pts = len(self.telemetry_history)
            dx = plot_w / float(self.telemetry_history.maxlen - 1)
            
            ear_points = []
            mar_points = []
            for i, (e_val, m_val, _) in enumerate(self.telemetry_history):
                px = int(plot_x + (plot_w - (n_pts - 1 - i) * dx))
                # Scale EAR: 0.0 to 0.45
                ey = int(plot_y + plot_h - (min(0.45, max(0.0, e_val)) / 0.45) * plot_h)
                # Scale MAR: 0.0 to 0.90
                my = int(plot_y + plot_h - (min(0.90, max(0.0, m_val)) / 0.90) * plot_h)
                ear_points.append((px, ey))
                mar_points.append((px, my))
            
            for i in range(len(ear_points) - 1):
                cv2.line(frame, ear_points[i], ear_points[i + 1], (0, 255, 120), 2)  # EAR = Green
                cv2.line(frame, mar_points[i], mar_points[i + 1], (0, 160, 255), 1)  # MAR = Orange
        
        # Draw Calibration Banner in screen center if calibrating
        if calibration_info and calibration_info.get('is_calibrating'):
            prog = calibration_info.get('progress', 0.0)
            rem = calibration_info.get('remaining', 0.0)
            
            cx, cy = w_frame // 2, h_frame // 2 - 40
            bw, bh = 460, 90
            bx, by = cx - bw // 2, cy - bh // 2
            
            cal_overlay = frame.copy()
            cv2.rectangle(cal_overlay, (bx, by), (bx + bw, by + bh), (15, 15, 15), -1)
            cv2.rectangle(cal_overlay, (bx, by), (bx + bw, by + bh), (0, 220, 255), 2)
            frame = cv2.addWeighted(cal_overlay, 0.85, frame, 0.15, 0)
            
            title_txt = f"CALIBRATING PERSONAL BASELINE ({rem:.1f}s)"
            cv2.putText(frame, title_txt, (bx + 20, by + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            cv2.putText(frame, "Please look naturally straight at the road", (bx + 20, by + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            
            # Progress bar
            bar_w = bw - 40
            bar_h = 12
            bar_x = bx + 20
            bar_y = by + 64
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * prog), bar_y + bar_h), (0, 220, 255), -1)
        
        # Draw alert icon if drowsy
        if state == self.STATE_DROWSY:
            # Flashing warning icon
            if int(time.time() * 2) % 2 == 0:  # Flash at 1 Hz
                icon_size = 60
                icon_x = w_frame - icon_size - 20
                icon_y = 20
                cv2.circle(frame, (icon_x, icon_y), icon_size // 2, (0, 0, 255), -1)
                cv2.putText(frame, "!", (icon_x - 15, icon_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        
        return frame
        
        return frame
    
    def get_state_name(self, state):
        """Get human-readable state name"""
        return ["ALERT", "WARNING", "DROWSY"][state]
    
    def get_statistics(self):
        """
        Get alert statistics from event history
        
        Returns:
            dict: Statistics dictionary
        """
        if not self.event_history:
            return {
                'total_events': 0,
                'alert_count': 0,
                'warning_count': 0,
                'drowsy_count': 0
            }
        
        alert_count = sum(1 for e in self.event_history if e['state'] == self.STATE_ALERT)
        warning_count = sum(1 for e in self.event_history if e['state'] == self.STATE_WARNING)
        drowsy_count = sum(1 for e in self.event_history if e['state'] == self.STATE_DROWSY)
        
        return {
            'total_events': len(self.event_history),
            'alert_count': alert_count,
            'warning_count': warning_count,
            'drowsy_count': drowsy_count
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.enable_audio:
            pygame.mixer.quit()
        
        if self.enable_voice and hasattr(self, 'tts_engine'):
            try:
                self.tts_engine.stop()
            except:
                pass
