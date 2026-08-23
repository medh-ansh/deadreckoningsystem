# import pandas as pd

# # Define the files to merge and their 'is_high' domain label
# files_to_merge = {
#     'train_data.csv': 0,             # Original low-speed data
#     'newtrain_data_low_grip.csv': 0, # Low-grip scenario
#     'newtrain_data_mirrored.csv': 0, # Mirrored track scenario
#     'newtrain_data_noisy.csv': 0,    # Bumpy track scenario          
#     'f1_high_speed_500k.csv': 1      # High-speed F1 dataset
# }

# dataframes = []

# for file_name, is_high_flag in files_to_merge.items():
#     print(f"Loading {file_name}...")
#     df = pd.read_csv(file_name)
    
#     # 1. Add the domain ID flag
#     df['is_high'] = is_high_flag
    
#     # 2. Drop time_ms if present to match schema
#     if 'time_ms' in df.columns:
#         df = df.drop(columns=['time_ms'])
        
#     dataframes.append(df)

# print("\nMerging all datasets sequentially (NO ROW SHUFFLING)...")
# # Stitching sequentially preserves continuous time-series blocks
# master_df = pd.concat(dataframes, ignore_index=True)

# output_filename = 'master_crnn_training_data.csv'
# print(f"Saving cleanly ordered dataset to {output_filename}...")
# master_df.to_csv(output_filename, index=False)

# print(f"\nSuccess! Master dataset re-created with {len(master_df)} continuous rows.")










import glob
import os
import pandas as pd

dataset_root = "Synchronised V abd S datasets/Categorised IOVNB Dataset"

# Define test driver folder (Driver D stays completely unseen during training)
test_driver_folder = "Y (Driver D)"

# Filter specifically for S-*.csv files!
search_pattern = os.path.join(dataset_root, "**", "S-*.csv")
s_csv_files = glob.glob(search_pattern, recursive=True)

print(f"Found {len(s_csv_files)} Sensor (S-*.csv) files to process.")

train_dfs = []
test_dfs = []

for file_path in s_csv_files:
    df = pd.read_csv(file_path, encoding='latin1').dropna()

    # Route Driver D into test set, all other drivers into training set
    if test_driver_folder in file_path:
        test_dfs.append(df)
    else:
        train_dfs.append(df)

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

# Export clean training and testing CSV files
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("final_test_data.csv", index=False)

print(
    f"\nSUCCESS: Created 'train_data.csv' with {len(train_df):,} rows from S-files."
)
print(
    f"SUCCESS: Created 'final_test_data.csv' with {len(test_df):,} rows from Driver D."
)














# import pandas as pd
# import numpy as np

# # 3,000,000 rows at 50ms intervals = ~41 hours of continuous driving
# n_rows = 3000000

# print("1. Generating continuous real-life speed curves...")
# # We use a cumulative random walk to simulate a car accelerating and decelerating over time
# raw_speed_noise = np.random.normal(0, 1, n_rows)
# speed_walk = np.cumsum(raw_speed_noise)

# # Normalize the speeds to exist strictly between 0 and 130 km/h
# speed_walk = speed_walk - np.min(speed_walk)
# speed_walk = (speed_walk / np.max(speed_walk)) * 130 

# # Apply a heavy rolling mean to simulate the heavy mass and smooth momentum of a street car
# smooth_speed = pd.Series(speed_walk).rolling(window=500, min_periods=1).mean().values

# print("2. Calculating realistic physics constraints...")
# # Derive longitudinal acceleration (Y-axis) directly from the speed changes
# # This guarantees that braking and accelerating perfectly match the target speed
# accel_y = np.gradient(smooth_speed) * 15.0 

# # Generate lateral dynamics (street cornering and lane changes)
# raw_steering = np.random.normal(0, 1, n_rows)
# steer_walk = np.cumsum(raw_steering)
# smooth_steering = pd.Series(steer_walk).rolling(window=500, min_periods=1).mean().values

# # Cap lateral Gs between -3.5 and +3.5 (Hard street driving, not F1 driving)
# smooth_steering = (smooth_steering / np.max(np.abs(smooth_steering))) * 3.5
# accel_x = smooth_steering

# # Yaw is directly proportional to cornering force
# gyro_yaw = accel_x * 0.1 

# # Add subtle road vibration to pitch, roll, and Z-accel (gravity)
# gyro_pitch = np.random.normal(0, 0.05, n_rows)
# gyro_roll = np.random.normal(0, 0.05, n_rows)
# accel_z = np.random.normal(9.8, 0.2, n_rows) 

# print("3. Building the dataset and applying domain logic...")
# df_real = pd.DataFrame({
#     'target_speed_kmh': smooth_speed,
#     'gyro_yaw': gyro_yaw,
#     'gyro_pitch': gyro_pitch,
#     'gyro_roll': gyro_roll,
#     'linear_acc_x': accel_x,
#     'linear_acc_y': accel_y,
#     'linear_acc_z': accel_z
# })

# # Dynamically set is_high to 1 ONLY when the street car exceeds 80 km/h
# df_real['is_high'] = (df_real['target_speed_kmh'] > 80).astype(int)

# output_filename = 'real_life_driving_3M.csv'
# print(f"4. Saving {n_rows} rows to {output_filename} (this might take a few seconds)...")
# df_real.to_csv(output_filename, index=False)
# print("Success! Dataset is ready.")