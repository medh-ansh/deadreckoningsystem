# import pandas as pd
# import numpy as np

# print("Loading original telemetry data...")
# df = pd.read_csv('train_data.csv')

# # --- SCENARIO 1: MIRRORED TRACK ---
# print("Generating Mirrored Track scenario...")
# df_mirrored = df.copy()
# df_mirrored['gyro_yaw'] *= -1
# df_mirrored['gyro_roll'] *= -1
# df_mirrored['linear_acc_x'] *= -1
# df_mirrored.to_csv('train_data_mirrored.csv', index=False)

# # --- SCENARIO 2: LOW GRIP / WET CONDITIONS ---
# print("Generating Low Grip scenario...")
# df_low_grip = df.copy()
# df_low_grip['target_speed_kmh'] *= 0.85
# df_low_grip['linear_acc_x'] *= 0.70  # 30% less lateral grip
# df_low_grip['linear_acc_y'] *= 0.80  # 20% less braking/accel efficiency
# df_low_grip.to_csv('train_data_low_grip.csv', index=False)

# # --- SCENARIO 3: BUMPY TRACK / NOISE INJECTION ---
# print("Generating Bumpy Track scenario...")
# df_noisy = df.copy()
# noise_level = 0.05 # 5% randomized noise

# for col in ['gyro_yaw', 'gyro_pitch', 'gyro_roll', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z']:
#     std_val = df_noisy[col].std()
#     # Generate Gaussian noise based on the column's natural variance
#     noise = np.random.normal(0, std_val * noise_level, size=len(df_noisy))
#     df_noisy[col] += noise
    
# df_noisy.to_csv('train_data_noisy.csv', index=False)

# print("Success! All new scenarios saved to your directory.")
import pandas as pd
import numpy as np

print("Loading base Hungary data...")
base_df = pd.read_csv('synthetic_f1_data_hungary.csv')

# Drop stationary pit-lane data so the model focuses entirely on racing speeds
base_df = base_df[base_df['target_speed_kmh'] > 50].copy()

def generate_track_data(df, speed_mult, lat_mult, long_mult, pitch_mult, rows_needed):
    # Sample randomly from the base data to build the required volume
    new_df = df.sample(n=rows_needed, replace=True).copy()
    
    # 1. Apply Track-Specific Physics
    new_df['target_speed_kmh'] = np.clip(new_df['target_speed_kmh'] * speed_mult, 0, 350)
    new_df['linear_acc_x'] *= lat_mult      # Lateral Gs (Cornering)
    new_df['linear_acc_y'] *= long_mult     # Longitudinal Gs (Braking/Accel)
    new_df['gyro_pitch'] *= pitch_mult      # Pitch (Elevation changes)
    
    # 2. Inject Gaussian Noise (prevents exact duplicates in the 500k rows)
    for col in ['gyro_yaw', 'gyro_pitch', 'gyro_roll', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z']:
        noise = np.random.normal(0, new_df[col].std() * 0.05, size=rows_needed)
        new_df[col] += noise
        
    return new_df

rows_per_circuit = 125000

print("Simulating Base Layout...")
df_base = generate_track_data(base_df, 1.0, 1.0, 1.0, 1.0, rows_per_circuit)

print("Simulating Elevation & Sweepers (e.g., Interlagos)...")
df_interlagos = generate_track_data(base_df, 1.02, 1.15, 0.95, 1.30, rows_per_circuit)

print("Simulating Heavy Braking & Straights (e.g., Yas Marina)...")
df_yas = generate_track_data(base_df, 1.08, 0.90, 1.25, 0.80, rows_per_circuit)

print("Simulating Extreme High Speed (e.g., Monza)...")
df_monza = generate_track_data(base_df, 1.12, 0.75, 1.10, 0.90, rows_per_circuit)

print("Merging into master dataset...")
final_df = pd.concat([df_base, df_interlagos, df_yas, df_monza]).reset_index(drop=True)

# Re-index the time_ms column so it flows sequentially from start to finish
final_df['time_ms'] = (np.arange(len(final_df)) * 50) + base_df['time_ms'].iloc[0]

# Save the massive dataset
output_filename = 'f1_high_speed_500k.csv'
final_df.to_csv(output_filename, index=False)
print(f"Success! Generated {len(final_df)} rows saved to {output_filename}.")