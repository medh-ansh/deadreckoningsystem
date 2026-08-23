import math
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from model_trainer import SpeedPredictorCRNN

# --- 1. Hardware & Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Engine Engaged: {device}")

model = SpeedPredictorCRNN().to(device)
model.load_state_dict(
    torch.load("trained_crnn_model.pth", map_location=device)
)
model.eval()

# Load the fitted scaler created during model_trainer.py
scaler = joblib.load("sensor_scaler.save")

# --- 2. Load Test Data ---
test_csv = "final_test_data.csv"
test_df = pd.read_csv(test_csv).dropna()

# Target speed (Column 0 in training format)
if "target_speed_kmh" in test_df.columns:
    true_speeds_kmh = test_df["target_speed_kmh"].values
else:
    true_speeds_kmh = test_df.iloc[:, 0].values

# Extract 7 IMU input features matching training data
feature_cols = [
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "is_high",
]
raw_features = test_df[feature_cols].values

# Extract yaw velocity for 2D heading updates
gyro_yaw_raw = test_df["gyro_yaw"].values

# Scale test features using saved scaler parameters
scaled_features = scaler.transform(raw_features)

# --- 3. Kinematic Dead Reckoning Setup ---
# IO-VNBD logs at 10Hz, so fixed dt = 0.10s (100ms)
dt = 0.10  
window_size = 50

x_true, y_true = [0.0], [0.0]
x_pred, y_pred = [0.0], [0.0]
current_heading = 0.0  # Radians

print("Running Fast AI Inference and Kinematic Engine...")

with torch.no_grad():
    for i in range(window_size, len(scaled_features)):
        # 50-step sequence window
        window = scaled_features[i - window_size : i]

        # Reshape to (batch_size=1, sequence_length=50, num_features=7)
        tensor_window = torch.FloatTensor(window).unsqueeze(0).to(device)

        # Predict speed via CRNN
        predicted_speed_kmh = model(tensor_window).item()
        actual_speed_kmh = true_speeds_kmh[i]

        # --- NOISE CLAMPING & DEADBAND FILTER ---
        # 1. Clamp negative raw predictions to 0.0
        v_pred_kmh = max(0.0, predicted_speed_kmh)
        v_true_kmh = max(0.0, actual_speed_kmh)

        # 2. Deadband filter: ignore sensor idle noise below 0.5 km/h
        if v_pred_kmh < 0.5:
            v_pred_ms = 0.0
        else:
            v_pred_ms = v_pred_kmh / 3.6

        if v_true_kmh < 0.5:
            v_true_ms = 0.0
        else:
            v_true_ms = v_true_kmh / 3.6

        # Update Heading (gyro_yaw in rad/s)
        gyro_z_rad = gyro_yaw_raw[i]
        current_heading += gyro_z_rad * dt

        # 2D Kinematic Position Updates (Meters)
        x_pred.append(
            x_pred[-1] + (v_pred_ms * math.cos(current_heading) * dt)
        )
        y_pred.append(
            y_pred[-1] + (v_pred_ms * math.sin(current_heading) * dt)
        )

        x_true.append(
            x_true[-1] + (v_true_ms * math.cos(current_heading) * dt)
        )
        y_true.append(
            y_true[-1] + (v_true_ms * math.sin(current_heading) * dt)
        )

print("Plotting Trajectory Map...")

# --- 4. Trajectory Visualization ---
plt.figure(figsize=(10, 8))
plt.plot(x_true, y_true, label="True GPS Path", color="blue", linewidth=2)
plt.plot(
    x_pred,
    y_pred,
    label="AI Dead Reckoning Path",
    color="red",
    linestyle="--",
    linewidth=2,
)
plt.title("IO-VNBD: 2D Trajectory Mapping in GPS-Denied Environment")
plt.xlabel("X Position (Meters)")
plt.ylabel("Y Position (Meters)")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Maintain 1:1 aspect ratio across axes
plt.show()