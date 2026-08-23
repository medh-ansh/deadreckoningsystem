import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# --- 1. CLEAN REAL-WORLD IMU DATASET ---
class IMUDataset(Dataset):

  def __init__(self, csv_file, window_size=50, is_training=True):
    print(f"Loading {csv_file} into memory...")
    try:
      df = pd.read_csv(csv_file, encoding="utf-8").dropna()
    except UnicodeDecodeError:
      df = pd.read_csv(csv_file, encoding="latin1").dropna()

    # Target speed (Column 0: target_speed_kmh)
    self.y = df.iloc[:, 0].values
    # 7 Features: linear_acc_x..z, gyro_yaw..roll, is_high
    X_raw = df.iloc[:, 1:].values
    self.window_size = window_size

    if is_training:
      self.scaler = StandardScaler()
      self.X = self.scaler.fit_transform(X_raw)
      joblib.dump(self.scaler, "sensor_scaler.save")
      print("Math scaler built and saved to 'sensor_scaler.save'!")
    else:
      self.scaler = joblib.load("sensor_scaler.save")
      self.X = self.scaler.transform(X_raw)

    self.X = torch.tensor(self.X, dtype=torch.float32)
    self.y = torch.tensor(self.y, dtype=torch.float32)

  def __len__(self):
    return len(self.X) - self.window_size

  def __getitem__(self, idx):
    # Pure clean window slice without synthetic noise additions
    window_X = self.X[idx : idx + self.window_size]
    target_y = self.y[idx + self.window_size - 1]
    return window_X, target_y


# --- 2. ADVANCED CRNN ARCHITECTURE ---
class SpeedPredictorCRNN(nn.Module):

  def __init__(self):
    super(SpeedPredictorCRNN, self).__init__()

    # Feature Extraction Block
    self.conv1 = nn.Conv1d(
        in_channels=7, out_channels=32, kernel_size=3, padding=1
    )
    self.relu = nn.ReLU()
    self.dropout1 = nn.Dropout(p=0.2)

    # 2-Layer Stacked GRU for Temporal Dynamics
    self.gru = nn.GRU(
        input_size=32,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        dropout=0.2,
    )

    # Linear Speed Prediction Layer
    self.fc1 = nn.Linear(64, 1)

  def forward(self, x):
    # Permute for 1D Conv: (Batch, Sequence, Features) -> (Batch, Features, Sequence)
    x = x.permute(0, 2, 1)
    x = self.conv1(x)
    x = self.relu(x)
    x = self.dropout1(x)
    x = x.permute(0, 2, 1)

    # Recurrent Temporal Integration
    out, _ = self.gru(x)
    last_memory_state = out[:, -1, :]

    # Output predicted speed
    final_speed = self.fc1(last_memory_state)
    return final_speed


# --- 3. HIGH-PERFORMANCE TRAINING ENGINE ---
if __name__ == "__main__":
  print("--- CRNN PIPELINE ---")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Hardware Engine Engaged: {device}")

  # Clean dataset initialization
  my_data = IMUDataset("train_data.csv", window_size=50, is_training=True)

  train_loader = DataLoader(
      my_data,
      batch_size=256,
      shuffle=True,
      pin_memory=True,
      num_workers=4,
  )

  model = SpeedPredictorCRNN().to(device)

  # Robust loss & AdamW with dynamic LR scheduling
  criterion = nn.HuberLoss(delta=1.0)
  optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode="min", factor=0.5, patience=1
  )

  epochs = 8
  print(
      f"\n--- STARTING TRAINING FOR {epochs} EPOCHS ---"
  )

  best_loss = float("inf")

  for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (features, targets) in enumerate(train_loader):
      features = features.to(device, non_blocking=True)
      targets = targets.to(device, non_blocking=True)

      optimizer.zero_grad(set_to_none=True)
      predictions = model(features)
      loss = criterion(predictions, targets.view(-1, 1))
      loss.backward()

      # Prevent gradient explosion
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

      optimizer.step()
      total_loss += loss.item()

      if batch_idx % 1000 == 0 and batch_idx > 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)
    print(f"\n--> End of Epoch {epoch+1} | Average Huber Loss: {avg_loss:.4f}")

    scheduler.step(avg_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current Learning Rate: {current_lr}\n")

    # Save model weights
    if avg_loss < best_loss:
      best_loss = avg_loss
      torch.save(model.state_dict(), "trained_crnn_model.pth")
      print(f"--> Saved best checkpoint to 'trained_crnn_model.pth'\n")

  print("CRNN Model training complete!")