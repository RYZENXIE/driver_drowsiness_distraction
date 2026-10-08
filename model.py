"""
Deep Neural Architecture for Driver Cognitive Distraction & Microsleep Detection
Dual Backend Support:
1. PyTorch DriverFatigueNet (when torch is available)
2. Scikit-Learn Deep Multi-Layer Perceptron (128 -> 64 -> 32) Neural Architecture
"""

import numpy as np

FEATURE_NAMES = [
    "ear",                    # Eye Aspect Ratio (0.15 - 0.38)
    "ear_velocity",           # Rate of eye closure (dEAR/dt)
    "perclos",                # Percentage of Eye Closure over window (0.0 - 1.0)
    "mar",                    # Mouth Aspect Ratio (0.10 - 0.90)
    "head_pitch",             # Pitch angle in degrees (-45° down to +45° up)
    "head_yaw",               # Yaw angle in degrees (-60° left to +60° right)
    "head_roll",              # Roll tilt angle (-30° to +30°)
    "blink_frequency",        # Blinks per minute
    "gaze_eccentricity",      # Deviation from road center line (0.0 - 1.0)
    "yawn_duration_sec"       # Duration of active mouth opening (sec)
]

CLASS_LABELS = {
    0: "ALERT",
    1: "DROWSY / MICROSLEEP",
    2: "DISTRACTED",
    3: "FATIGUED / YAWNING"
}

CLASS_COLORS = {
    0: "#00E676",  # Green
    1: "#FF1744",  # Red Alert
    2: "#FF9100",  # Orange Warning
    3: "#FFD600"   # Yellow Caution
}

# Optional PyTorch Definition
try:
    import torch
    import torch.nn as nn
    
    class DriverFatigueNet(nn.Module):
        def __init__(self, input_dim=10, num_classes=4):
            super(DriverFatigueNet, self).__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            self.classifier = nn.Linear(32, num_classes)
            self.fatigue_regressor = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            feats = self.feature_extractor(x)
            logits = self.classifier(feats)
            fatigue_score = self.fatigue_regressor(feats)
            return logits, fatigue_score

except ImportError:
    DriverFatigueNet = None
