from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

main_dir = Path("D:/sih/Synchronised V abd S datasets/Categorised IOVNB Dataset")

# Train on Drivers A, B, and E; Test on Driver D (Unseen Driver Split)
train_folders = [
    "S (Driver A)",
    "M (Driver B)",
    "Vf (Driver E)",
    "Vta (Driver E)",
    "Vtb (Driver E)",
    "Vw (Driver E)",
]
test_folders = ["Y (Driver D)"]


def calculate_gravity(latitude_degrees, altitude_meters):
  lat_rad = np.radians(latitude_degrees)

  g_e = 9.7803253359
  k = 0.0019318526524
  e_sq = 0.0066943799901

  sin_lat_sq = np.sin(lat_rad) ** 2
  num = 1 + (k * sin_lat_sq)
  den = np.sqrt(1 - (e_sq * sin_lat_sq))

  gsea = g_e * (num / den)
  g_local = gsea - (0.000003086 * altitude_meters)

  return g_local


def remove_gravity(g_val, dataframe):
  earth_gravity = np.array([0, 0, g_val])
  yaw = dataframe["orient_yaw"].values
  pitch = dataframe["orient_pitch"].values
  roll = dataframe["orient_roll"].values

  euler_angles = np.column_stack((yaw, pitch, roll))
  rotations = Rotation.from_euler("ZYX", euler_angles, degrees=True)
  local_gravity_vectors = rotations.apply(earth_gravity)

  dataframe["linear_acc_x"] = dataframe["acc_x"] - local_gravity_vectors[:, 0]
  dataframe["linear_acc_y"] = dataframe["acc_y"] - local_gravity_vectors[:, 1]
  dataframe["linear_acc_z"] = dataframe["acc_z"] - local_gravity_vectors[:, 2]

  # Drop raw accelerations & orientations, but keep gravity removed linear components
  return dataframe.drop(
      columns=[
          "acc_x",
          "acc_y",
          "acc_z",
          "orient_yaw",
          "orient_pitch",
          "orient_roll",
      ]
  )


def process(folder_list, output_filename):
  all_dataframes = []

  for folder_name in folder_list:
    folder_path = main_dir / folder_name
    # Recursive search to catch subfolders like Y1, Y2, etc.
    csv_files = list(folder_path.rglob("S-*.csv"))

    for file in csv_files:
      print(f"Loading {file.name} from {folder_name}...")
      try:
        df = pd.read_csv(file, encoding="utf-8")
      except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="latin1")

      df.columns = df.columns.str.strip()

      start_lat = df["GPS LATITUDE (degrees)"].iloc[0]
      start_alt = df["GPS ALTITUDE (m)"].iloc[0]
      g_local = calculate_gravity(start_lat, start_alt)

      col_mapping = {}
      for col in df.columns:
        if "TIME SINCE START" in col:
          col_mapping[col] = "time_ms"
        elif "GPS SPEED" in col:
          col_mapping[col] = "target_speed_kmh"
        elif "ACCELEROMETER X" in col:
          col_mapping[col] = "acc_x"
        elif "ACCELEROMETER Y" in col:
          col_mapping[col] = "acc_y"
        elif "ACCELEROMETER Z" in col:
          col_mapping[col] = "acc_z"
        elif "GYROSCOPE Yaw" in col:
          col_mapping[col] = "gyro_yaw"
        elif "GYROSCOPE Pitch" in col:
          col_mapping[col] = "gyro_pitch"
        elif "GYROSCOPE Roll" in col:
          col_mapping[col] = "gyro_roll"
        elif "ORIENTATION (Yaw)" in col:
          col_mapping[col] = "orient_yaw"
        elif "ORIENTATION (Pitch)" in col:
          col_mapping[col] = "orient_pitch"
        elif "ORIENTATION (Roll" in col:
          col_mapping[col] = "orient_roll"

      df_clean = df[list(col_mapping.keys())].copy()
      df_clean = df_clean.rename(columns=col_mapping)

      # Perform coordinate transform and remove local gravity vector
      df_clean = remove_gravity(g_local, df_clean)

      # Create is_high categorical flag (1 if speed > 50 km/h, else 0)
      df_clean["is_high"] = (df_clean["target_speed_kmh"] > 50.0).astype(int)

      # Enforce EXACT column sequence required by model_trainer.py
      final_columns = [
          "target_speed_kmh",
          "linear_acc_x",
          "linear_acc_y",
          "linear_acc_z",
          "gyro_yaw",
          "gyro_pitch",
          "gyro_roll",
          "is_high",
      ]

      df_clean = df_clean[final_columns].dropna()
      all_dataframes.append(df_clean)

  print(f"\nFound {len(all_dataframes)} processed CSV files for {folder_list}.")
  if len(all_dataframes) > 0:
    print(f"Stitching together for {output_filename}...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    master_df.to_csv(output_filename, index=False)
    print(f"Successfully saved {output_filename} ({len(master_df):,} rows)!\n")
  else:
    print(f"Error: No dataframes were found for {output_filename}!")


process(train_folders, "train_data.csv")
process(test_folders, "final_test_data.csv")

print("Processing complete! Ready for model_trainer.py.")