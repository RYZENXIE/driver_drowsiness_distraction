"""
Deep Learning Training Pipeline for Driver Fatigue & Distraction
Trains:
1. Deep Multi-Layer Neural Network (Input 10 -> Dense 128 -> Dense 64 -> Dense 32 -> Output 4)
2. Continuous Fatigue Severity Regressor
3. PyTorch DriverFatigueNet checkpoint (if PyTorch installed)
Evaluates Accuracy, Loss, Precision, Recall and exports saved models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import FEATURE_NAMES, CLASS_LABELS

def generate_driver_dataset(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    n_per_class = n_samples // 4
    records = []
    
    # 0: ALERT
    for _ in range(n_per_class):
        ear = np.random.normal(0.31, 0.03)
        ear_vel = np.random.normal(0.0, 0.02)
        perclos = np.random.beta(1.5, 10.0) * 0.18
        mar = np.random.normal(0.18, 0.04)
        pitch = np.random.normal(2.0, 4.0)
        yaw = np.random.normal(0.0, 6.0)
        roll = np.random.normal(0.0, 3.0)
        blinks = np.random.normal(16.0, 3.5)
        gaze_ecc = np.random.beta(1.0, 8.0) * 0.25
        yawn_dur = 0.0
        records.append([ear, ear_vel, perclos, mar, pitch, yaw, roll, blinks, gaze_ecc, yawn_dur, 0, 0.08])

    # 1: DROWSY / MICROSLEEP
    for _ in range(n_per_class):
        ear = np.random.normal(0.15, 0.03)
        ear_vel = np.random.normal(-0.06, 0.03)
        perclos = np.random.beta(6.0, 2.0) * 0.55 + 0.35
        mar = np.random.normal(0.22, 0.05)
        pitch = np.random.normal(-18.0, 6.0)
        yaw = np.random.normal(0.0, 8.0)
        roll = np.random.normal(4.0, 6.0)
        blinks = np.random.normal(6.0, 2.5)
        gaze_ecc = np.random.uniform(0.1, 0.6)
        yawn_dur = np.random.choice([0.0, 1.2, 2.5], p=[0.6, 0.3, 0.1])
        records.append([ear, ear_vel, perclos, mar, pitch, yaw, roll, blinks, gaze_ecc, yawn_dur, 1, 0.88])

    # 2: DISTRACTED
    for _ in range(n_per_class):
        ear = np.random.normal(0.30, 0.03)
        ear_vel = np.random.normal(0.0, 0.02)
        perclos = np.random.beta(2.0, 8.0) * 0.20
        mar = np.random.normal(0.20, 0.04)
        if np.random.rand() > 0.45:
            yaw = np.random.choice([-1, 1]) * np.random.uniform(26.0, 52.0)
            pitch = np.random.normal(0.0, 6.0)
        else:
            yaw = np.random.normal(0.0, 8.0)
            pitch = np.random.uniform(-38.0, -22.0)
        roll = np.random.normal(0.0, 5.0)
        blinks = np.random.normal(15.0, 4.0)
        gaze_ecc = np.random.uniform(0.55, 0.95)
        yawn_dur = 0.0
        records.append([ear, ear_vel, perclos, mar, pitch, yaw, roll, blinks, gaze_ecc, yawn_dur, 2, 0.52])

    # 3: FATIGUED / YAWNING
    for _ in range(n_per_class):
        ear = np.random.normal(0.22, 0.04)
        ear_vel = np.random.normal(-0.02, 0.03)
        perclos = np.random.beta(3.0, 5.0) * 0.35 + 0.10
        mar = np.random.uniform(0.52, 0.88)
        pitch = np.random.normal(8.0, 8.0)
        yaw = np.random.normal(0.0, 8.0)
        roll = np.random.normal(0.0, 4.0)
        blinks = np.random.normal(24.0, 5.0)
        gaze_ecc = np.random.uniform(0.15, 0.50)
        yawn_dur = np.random.uniform(2.2, 5.5)
        records.append([ear, ear_vel, perclos, mar, pitch, yaw, roll, blinks, gaze_ecc, yawn_dur, 3, 0.72])

    cols = FEATURE_NAMES + ["class_label", "fatigue_severity"]
    df = pd.DataFrame(records, columns=cols)
    df["ear"] = np.clip(df["ear"], 0.05, 0.45)
    df["perclos"] = np.clip(df["perclos"], 0.0, 1.0)
    df["mar"] = np.clip(df["mar"], 0.08, 0.95)
    df["gaze_eccentricity"] = np.clip(df["gaze_eccentricity"], 0.0, 1.0)
    df["fatigue_severity"] = np.clip(df["fatigue_severity"], 0.0, 1.0)
    return df

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_models_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    data_path = os.path.join(base_dir, "driver_telemetry.csv")
    print("Generating 5,000 driving telemetry frames (NHTSA benchmarks)...")
    df = generate_driver_dataset(5000)
    df.to_csv(data_path, index=False)
    
    X = df[FEATURE_NAMES].values
    y_class = df["class_label"].values
    y_score = df["fatigue_severity"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_cls_train, y_cls_test, y_scr_train, y_scr_test = train_test_split(
        X_scaled, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print("\n--- Training Deep Neural Network (Architecture: 10 -> 128 -> 64 -> 32 -> 4) ---")
    deep_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=150,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    deep_clf.fit(X_train, y_cls_train)
    
    deep_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=150,
        random_state=42
    )
    deep_reg.fit(X_train, y_scr_train)
    
    y_pred = deep_clf.predict(X_test)
    acc = accuracy_score(y_cls_test, y_pred)
    print(f"\n[OK] Deep Neural Network Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    target_names = [CLASS_LABELS[i] for i in range(4)]
    print(classification_report(y_cls_test, y_pred, target_names=target_names))
    
    # Save Scikit-Learn / Joblib Neural Models
    clf_path = os.path.join(saved_models_dir, "deep_fatigue_classifier.joblib")
    reg_path = os.path.join(saved_models_dir, "deep_fatigue_regressor.joblib")
    scaler_path = os.path.join(saved_models_dir, "telemetry_scaler.joblib")
    
    joblib.dump(deep_clf, clf_path)
    joblib.dump(deep_reg, reg_path)
    joblib.dump(scaler, scaler_path)

    # Optional PyTorch training if torch is present
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from model import DriverFatigueNet
        
        print("\nPyTorch detected! Exporting PyTorch DriverFatigueNet checkpoint...")
        pt_model = DriverFatigueNet(input_dim=len(FEATURE_NAMES), num_classes=4)
        pt_train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_cls_train, dtype=torch.long),
            torch.tensor(y_scr_train, dtype=torch.float32).unsqueeze(1)
        )
        pt_loader = DataLoader(pt_train_dataset, batch_size=64, shuffle=True)
        pt_optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.003)
        pt_criterion = nn.CrossEntropyLoss()
        
        pt_model.train()
        for _ in range(25):
            for bx, by, _ in pt_loader:
                pt_optimizer.zero_grad()
                logits, _ = pt_model(bx)
                loss = pt_criterion(logits, by)
                loss.backward()
                pt_optimizer.step()
                
        torch.save(pt_model.state_dict(), os.path.join(saved_models_dir, "driver_fatigue_net.pt"))
        print("[OK] PyTorch checkpoint successfully saved!")
    except ImportError:
        pass
        
    meta = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
        "class_labels": CLASS_LABELS,
        "test_accuracy": round(float(acc), 4),
        "architecture": "Deep Neural Network: Input(10) -> Dense(128) -> Dense(64) -> Dense(32) -> Output(4)"
    }
    with open(os.path.join(saved_models_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"\nAll models and metadata saved to: {saved_models_dir}")

if __name__ == "__main__":
    train_model()
