# import socket
# import csv
# import time
# from f1_2020_telemetry.packets import unpack_udp_packet

# # --- 1. SETUP THE CSV FILE ---
# csv_filename = "synthetic_f1_data_hungary.csv"
# headers = [
#     "target_speed_kmh", "time_ms", 
#     "gyro_yaw", "gyro_pitch", "gyro_roll", 
#     "linear_acc_x", "linear_acc_y", "linear_acc_z"
# ]

# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)

# # --- 2. SETUP THE UDP LISTENER ---
# UDP_IP = "127.0.0.1"
# UDP_PORT = 20777
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((UDP_IP, UDP_PORT))

# sock.settimeout(1.0)    

# print(f"Listening for F1 2020 Telemetry on {UDP_IP}:{UDP_PORT}...")
# print("Drive a few laps! Press Ctrl+C in this terminal when you are done.")

# # --- 3. THE DATA LOOP ---
# current_speed = 0.0
# start_time = time.time()

# try:
#     with open(csv_filename, mode='a', newline='') as file:
#         writer = csv.writer(file)
        
#         while True:
#             try:
#                 data, addr = sock.recvfrom(2048)
#             except socket.timeout:
#                 continue # If we hear nothing for 1 second, just loop back and try again!
            
#             packet = unpack_udp_packet(data)
            
#             # F1 2020 sends different packets for different info. 
#             # Packet ID 6 is Car Telemetry (Speed)
#             if packet.header.packetId == 6:
#                 # Get the speed of Car 0 (the player)
#                 current_speed = packet.carTelemetryData[packet.header.playerCarIndex].speed
                
#             # Packet ID 0 is Motion (G-forces, Yaw, Pitch, Roll)
#             elif packet.header.packetId == 0:
#                 motion = packet.carMotionData[packet.header.playerCarIndex]
                
#                 # Calculate elapsed time in milliseconds
#                 time_ms = int((time.time() - start_time) * 1000)
                
#                 # Write the combined row to our CSV
#                 writer.writerow([
#                     current_speed,
#                     time_ms,
#                     motion.yaw,
#                     motion.pitch,
#                     motion.roll,
#                     motion.gForceLateral,      # Maps to X
#                     motion.gForceLongitudinal, # Maps to Y
#                     motion.gForceVertical      # Maps to Z
#                 ])

#                 file.flush() # Assuming your file object is named csv_file

# except KeyboardInterrupt:
#     print(f"\nData collection stopped. Saved to {csv_filename}!")

import pandas as pd

# Load your master dataset
df = pd.read_csv("master_crnn_training_data.csv")

# Filter out rows where target speed exceeds 150 km/h
df_capped = df[df.iloc[:, 0] <= 150.0]

# Save the calibrated dataset
df_capped.to_csv("master_crnn_training_data_150kmh.csv", index=False)
print(f"Original rows: {len(df)} | Capped dataset rows: {len(df_capped)}")

