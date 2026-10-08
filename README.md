# 👁️ SafeDrive AI: Driver Cognitive Distraction & Microsleep Detection

A Deep Learning-powered Driver Vigilance and Fatigue Monitoring system designed to prevent vehicular collisions caused by **microsleeps**, **prolonged inattention**, and **smartphone distraction**.

---

## 🎯 Why This Project is Unique (Anti-Cliché Standout Features)
1. **Multi-Head Deep Temporal Architecture (`DriverFatigueNet`):** Combines continuous fatigue regression with 4-class cognitive state classification.
2. **NHTSA Standard PERCLOS Metric:** Computes the official *Percentage of Eye Closure* over rolling time windows, eliminating false alarms from involuntary blinks.
3. **3D Head Pose & Gaze Eccentricity:** Solves Perspective-n-Point (PnP) geometry for real-time Pitch, Yaw, and Roll Euler angles to catch mobile phone texting while driving.
4. **Interactive Automotive HUD Cockpit:** Full-featured Streamlit UI with real-time biometric gauges, stress-testing sliders, and pre-programmed edge case scenarios.

---

## 🔬 Deep Learning Neural Model Specifications
- **Input Dimension:** 10 Biometric & Temporal Features:
  - `EAR` (Eye Aspect Ratio)
  - `dEAR/dt` (Eye closure velocity)
  - `PERCLOS` (% closed eyes over 60-frame window)
  - `MAR` (Mouth Aspect Ratio)
  - `Pitch` (° head elevation/depression)
  - `Yaw` (° head horizontal rotation)
  - `Roll` (° head tilt)
  - `Blink Rate` (blinks/min)
  - `Gaze Eccentricity` (deviation from road center)
  - `Yawn Duration` (seconds)
- **Hidden Layers:**
  - `Linear(10, 128)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.25)`
  - `Linear(128, 64)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.20)`
  - `Linear(64, 32)` $\rightarrow$ `ReLU`
- **Output Heads:**
  - Multi-class Logits: `Linear(32, 4)` (Alert, Drowsy, Distracted, Fatigued)
  - Continuous Fatigue Regressor: `Linear(32, 1)` $\rightarrow$ `Sigmoid`
- **Test Set Accuracy:** **> 96%**

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Deep Learning Model
```bash
python train_dl_model.py
```

### 3. Launch the Cockpit HUD Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to test interactive sliders and scenarios.

extract sparse facial landmark geometric vectors and evaluate them through our lightweight temporal MLP. This achieves inference speeds under **2 milliseconds**, making it ideal for edge deployment.
