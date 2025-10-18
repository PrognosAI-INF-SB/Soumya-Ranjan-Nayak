# ===============================
# Milestone 2: LSTM Model Training
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load Processed Data (from Milestone 1)
# -------------------------------
train = pd.read_csv("../milestone_1_data_preparation/processed_data/train_processed_FD001.csv")
test = pd.read_csv("../milestone_1_data_preparation/processed_data/test_processed_FD001.csv")
rul = pd.read_csv("../milestone_1_data_preparation/processed_data/rul_targets_FD001.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("RUL shape:", rul.shape)

# -------------------------------
# Function to create LSTM sequences
# -------------------------------
def create_sequences(data, sequence_length=30):
    sequences = []
    targets = []
    for engine_id in data['engine_id'].unique():
        engine_data = data[data['engine_id']==engine_id].sort_values('cycle')
        sensor_cols = [c for c in engine_data.columns if 'sensor' in c or 'op_setting' in c]
        for i in range(len(engine_data) - sequence_length + 1):
            seq = engine_data[sensor_cols].iloc[i:i+sequence_length].values
            target = engine_data['RUL'].iloc[i+sequence_length-1]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)

# Create sequences
sequence_length = 30
X_train, y_train = create_sequences(train, sequence_length)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# -------------------------------
# Define LSTM Model
# -------------------------------
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)

# -------------------------------
# Save Model
# -------------------------------
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/lstm_rul_model_FD001.h5")
print("Model saved in saved_models/lstm_rul_model_FD001.h5")

# -------------------------------
# Plot Training Loss
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()
