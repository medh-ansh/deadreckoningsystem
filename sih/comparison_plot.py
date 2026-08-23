import math
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import torch
from model_trainer import SpeedPredictorCRNN  # Your model file

# --- 1. SETUP & MODEL LOADING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeedPredictorCRNN().to(device)
model.load_state_dict(
    torch.load("trained_crnn_model.pth", map_location=device)
)
model.eval()

scaler = joblib.load("sensor_scaler.save")
test_df = pd.read_csv("final_test_data.csv").dropna()

# --- 2. TRAJECTORY CALCULATIONS ---
# Pre-allocate for speed
true_speeds_kmh = test_df["target_speed_kmh"].values
# Raw values for standard physics baseline
acc_x_raw = test_df["linear_acc_x"].values
acc_y_raw = test_df["linear_acc_y"].values
gyro_yaw_raw = test_df["gyro_yaw"].values

feature_cols = [
    "linear_acc_x", "linear_acc_y", "linear_acc_z",
    "gyro_yaw", "gyro_pitch", "gyro_roll", "is_high"
]
scaled_features = scaler.transform(test_df[feature_cols].values)

dt = 0.10  # 10 Hz interval
window_size = 50

# Track Initialization
# Path A: True GPS
x_true, y_true = [0.0], [0.0]
# Path B: AI CRNN Dead Reckoning
x_crnn, y_crnn = [0.0], [0.0]
# Path C: Naive Double Integration (Standard Physics)
x_naive, y_naive = [0.0], [0.0]
v_x_naive, v_y_naive = 0.0, 0.0

heading_crnn = 0.0

with torch.no_grad():
  for i in range(window_size, len(scaled_features)):
    # --- AI CRNN Path Calculation ---
    window = scaled_features[i - window_size : i]
    tensor_window = torch.FloatTensor(window).unsqueeze(0).to(device)

    pred_kmh = max(0.0, model(tensor_window).item())
    true_kmh = max(0.0, true_speeds_kmh[i])

    v_crnn_ms = 0.0 if pred_kmh < 0.5 else pred_kmh / 3.6
    v_true_ms = 0.0 if true_kmh < 0.5 else true_kmh / 3.6

    heading_crnn += gyro_yaw_raw[i] * dt

    x_crnn.append(x_crnn[-1] + (v_crnn_ms * math.cos(heading_crnn) * dt))
    y_crnn.append(y_crnn[-1] + (v_crnn_ms * math.sin(heading_crnn) * dt))

    x_true.append(x_true[-1] + (v_true_ms * math.cos(heading_crnn) * dt))
    y_true.append(y_true[-1] + (v_true_ms * math.sin(heading_crnn) * dt))

    # --- Naive Double Integration (Standard Physics) ---
    v_x_naive += acc_x_raw[i] * dt
    v_y_naive += acc_y_raw[i] * dt

    x_naive.append(x_naive[-1] + v_x_naive * dt)
    y_naive.append(y_naive[-1] + v_y_naive * dt)


# --- 3. CLEAN SINGLE-GRAPH A/B COMPARISON ---
plt.figure(figsize=(12, 9))

# Plot True GPS and AI CRNN Path
plt.plot(x_true, y_true, label="True GPS Path", color="blue", linewidth=2.5, zorder=3)
plt.plot(x_crnn, y_crnn, label="AI CRNN Dead Reckoning", color="red", linestyle="--", linewidth=2.5, zorder=3)

# Truncate Naive Integration to first 30 seconds (300 steps) so it doesn't break the axis limits
x_naive_truncated = x_naive[:350]
y_naive_truncated = y_naive[:350]

plt.plot(
    x_naive_truncated,
    y_naive_truncated,
    label="Naive Double Integration (First 35s Drift)",
    color="black",
    linestyle=":",
    linewidth=2,
    zorder=2
)

# Add Arrow pointing out the exploding drift trajectory
# plt.annotate(
#     "Naive Physics Explodes Off-Screen\n(Drifts >100,000m in minutes)",
#     xy=(x_naive_truncated[-1], y_naive_truncated[-1]),
#     xytext=(x_naive_truncated[-1] - 3000, y_naive_truncated[-1] + 1500),
#     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
#     fontsize=10,
#     fontweight='bold',
#     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1)
# )

# Adjust the text box position (xytext) slightly to the right
plt.annotate(
    "Naive Physics Explodes Off-Screen\n(Drifts >100,000m in minutes)",
    xy=(x_naive_truncated[-1], y_naive_truncated[-1]),
    xytext=(
        x_naive_truncated[-1] + 500,
        y_naive_truncated[-1] + 800,
    ),  # Moved inside visible frame
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    fontsize=10,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
)

# Expand the left X-axis limit so the text box has breathing room
plt.xlim(-6000, 6000)

plt.title("A/B Validation: AI Dead Reckoning vs. Naive Double Integration Drift", fontsize=13, fontweight="bold")
plt.xlabel("X Position (Meters)", fontsize=11)
plt.ylabel("Y Position (Meters)", fontsize=11)

# Focus coordinate window purely on the true route boundary
plt.xlim(-3000, 6000)
plt.ylim(-2000, 5500)

plt.legend(loc="upper right", fontsize=11, framealpha=0.9)
plt.grid(True, linestyle="--", alpha=0.6)
plt.axis("equal")  # Preserve 1:1 spatial aspect ratio

plt.tight_layout()
plt.show()