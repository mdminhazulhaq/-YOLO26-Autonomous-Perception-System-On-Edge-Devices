import pandas as pd

path = r"C:\Users\UM-User\Downloads\training_lidar_box_10023947602400723454_1120_000_1140_000.parquet"

df = pd.read_parquet(path)

print("\n========== COLUMN NAMES ==========\n")

print(df.columns)

print("\n========== FIRST 10 ROWS ==========\n")

print(df.head(10))