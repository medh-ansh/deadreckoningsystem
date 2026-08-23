import pandas as pd

df = pd.read_csv("train_data.csv")

# Print top 5 lowest speed rows (unrounded full floats)
print("--- Lowest Speed Entries ---")
print(df["target_speed_kmh"].nsmallest(5).to_string())

# Print top 5 highest speed rows
print("\n--- Highest Speed Entries ---")
print(df["target_speed_kmh"].nlargest(5).to_string())