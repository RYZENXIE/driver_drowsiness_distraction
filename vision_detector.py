"""
Facial Geometry & Biometric Telemetry Extraction Engine
Computes:
1. Eye Aspect Ratio (EAR) & PERCLOS (Percentage of Eye Closure)
2. Mouth Aspect Ratio (MAR) & Yawn Detection
3. 3D Head Pose Angles (Yaw, Pitch, Roll)
4. Gaze Fixation & Blink Velocity Tracking
"""

import numpy as np
import collections

class DriverTelemetryEngine:
    def __init__(self, history_len=60, ear_threshold=0.20, mar_threshold=0.55):
        self.history_len = history_len
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        
        # Sliding temporal buffers
        self.ear_history = collections.deque(maxlen=history_len)
        self.mar_history = collections.deque(maxlen=history_len)
        self.pitch_history = collections.deque(maxlen=history_len)
        self.yaw_history = collections.deque(maxlen=history_len)
        
        self.blink_count = 0
        self.in_blink = False
        self.yawn_frames = 0
        self.prev_ear = 0.32

    def calculate_ear(self, eye_points):
        """
        Calculates Eye Aspect Ratio (EAR) based on Soukupova & Cech formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        p1, p2, p3, p4, p5, p6 = [np.array(p) for p in eye_points]
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        if h == 0:
            return 0.30
        return (v1 + v2) / (2.0 * h)

    def calculate_mar(self, mouth_points):
        """
        Calculates Mouth Aspect Ratio (MAR) to detect yawning:
        MAR = (||top - bottom||) / (||left - right||)
        """
        # Outer lip landmarks
        p_left, p_right, p_top, p_bottom = [np.array(p) for p in mouth_points[:4]]
        v = np.linalg.norm(p_top - p_bottom)
        h = np.linalg.norm(p_left - p_right)
        if h == 0:
            return 0.20
        return v / h

    def update_frame(self, current_ear, current_mar, head_pitch, head_yaw, head_roll):
        """
        Pushes a new frame observation into sliding buffer and computes
        temporal metrics (velocity, PERCLOS, blink rate).
        """
        self.ear_history.append(current_ear)
        self.mar_history.append(current_mar)
        self.pitch_history.append(head_pitch)
        self.yaw_history.append(head_yaw)
        
        # EAR velocity (dEAR / dt)
        ear_vel = current_ear - self.prev_ear
        self.prev_ear = current_ear
        
        # PERCLOS: % of time eyes were closed (EAR < threshold) over sliding window
        closed_frames = sum(1 for e in self.ear_history if e < self.ear_threshold)
        perclos = closed_frames / max(1, len(self.ear_history))
        
        # Blink tracking
        if current_ear < self.ear_threshold:
            if not self.in_blink:
                self.in_blink = True
                self.blink_count += 1
        else:
            self.in_blink = False
            
        # Estimated blink rate per minute (assuming ~30 fps)
        fps = 30.0
        window_sec = len(self.ear_history) / fps
        blink_freq = (self.blink_count / max(1.0, window_sec)) * 60.0
        blink_freq = min(60.0, blink_freq)
        
        # Yawn tracking
        if current_mar > self.mar_threshold:
            self.yawn_frames += 1
        else:
            self.yawn_frames = max(0, self.yawn_frames - 2)
        yawn_duration_sec = self.yawn_frames / fps
        
        # Gaze eccentricity (derived from yaw and pitch deviation from center)
        norm_yaw = abs(head_yaw) / 60.0
        norm_pitch = abs(head_pitch) / 45.0
        gaze_eccentricity = min(1.0, np.sqrt(norm_yaw**2 + norm_pitch**2))
        
        # Returns the 10-dimensional feature vector matching DriverFatigueNet
        features = [
            current_ear,
            ear_vel,
            perclos,
            current_mar,
            head_pitch,
            head_yaw,
            head_roll,
            blink_freq,
            gaze_eccentricity,
            yawn_duration_sec
        ]
        return np.array(features, dtype=np.float32)

    def reset_buffers(self):
        self.ear_history.clear()
        self.mar_history.clear()
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.blink_count = 0
        self.yawn_frames = 0
