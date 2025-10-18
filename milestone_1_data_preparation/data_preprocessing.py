import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
train_path = "../data/train_FD001.txt"
test_path = "../data/test_FD001.txt"
rul_path = "../data/RUL_FD001.txt"

# NASA dataset has no headers and extra spaces → handle that
def load_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.dropna(axis=1, how='all', inplace=True)
    return df

train_df = load_data(train_path)
test_df = load_data(test_path)
rul_df = pd.read_csv(rul_path, header=None)

# -----------------------------
# 2. Assign Column Names
# -----------------------------
col_names = ['engine_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
train_df.columns = col_names
test_df.columns = col_names

# -----------------------------
# 3. Compute RUL (Remaining Useful Life)
# -----------------------------
def compute_rul(df):
    rul = df.groupby('engine_id')['cycle'].max().reset_index()
    rul.columns = ['engine_id', 'max_cycle']
    df = df.merge(rul, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

train_df = compute_rul(train_df)

# -----------------------------
# 4. Normalize Sensor Values
# -----------------------------
sensor_cols = [c for c in train_df.columns if 'sensor' in c]
scaler = MinMaxScaler()
train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

# -----------------------------
# 5. Save Cleaned Data
# -----------------------------
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(f"{output_dir}/train_processed_FD001.csv", index=False)
test_df.to_csv(f"{output_dir}/test_processed_FD001.csv", index=False)

rul_df.columns = ['RUL']
rul_df.to_csv(f"{output_dir}/rul_targets_FD001.csv", index=False)

print("✅ Preprocessing complete. Files saved in 'processed_data/'")
