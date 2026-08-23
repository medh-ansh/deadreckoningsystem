import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from model_trainer import SpeedPredictorCRNN

# Load model and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeedPredictorCRNN().to(device)
model.load_state_dict(
    torch.load("trained_crnn_model.pth", map_location=device)
)
model.eval()

scaler = joblib.load("sensor_scaler.save")

# Load unseen test set (Driver D)
test_df = pd.read_csv("final_test_data.csv").dropna()
true_speeds_kmh = test_df["target_speed_kmh"].values

feature_cols = [
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "is_high",
]
scaled_features = scaler.transform(test_df[feature_cols].values)

window_size = 50
predictions = []

with torch.no_grad():
  for i in range(window_size, len(scaled_features)):
    window = scaled_features[i - window_size : i]
    tensor_window = torch.FloatTensor(window).unsqueeze(0).to(device)

    pred_kmh = model(tensor_window).item()
    # Apply zero-clamp filter for speed evaluation
    predictions.append(max(0.0, pred_kmh))

# Align ground truth targets with the sequence window shift
targets = true_speeds_kmh[window_size:]

# Calculate error metrics
mae = mean_absolute_error(targets, predictions)
rmse = np.sqrt(mean_squared_error(targets, predictions))
max_err = np.max(np.abs(targets - predictions))

print(f"--- MODEL ERROR METRICS ON UNSEEN TEST DRIVER (Driver D) ---")
print(f"Mean Absolute Error (MAE) : {mae:.3f} km/h")
print(f"Root Mean Squared Error   : {rmse:.3f} km/h")
print(f"Maximum Pointwise Error   : {max_err:.3f} km/h")