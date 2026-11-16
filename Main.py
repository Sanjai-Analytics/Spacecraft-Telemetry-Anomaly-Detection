import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt

# 1. Simulate pandas telemetry data loading (Replace with NASA MSL/SMAP data)
# Assume 'df' has a 'timestamp' index and a 'voltage' column
np.random.seed(42)
time_steps = 3000
df = pd.DataFrame({
    'voltage': np.sin(np.linspace(0, 50, time_steps)) + np.random.normal(0, 0.1, time_steps)
})
# Inject a synthetic anomaly for testing
df.iloc[1500:1550, 0] += 2.0 

# 2. Sequence Generator for Time Series (Sliding Window)
def create_sequences(values, time_steps=30):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

# Standardize the data
train_mean = df['voltage'].mean()
train_std = df['voltage'].std()
df['voltage_scaled'] = (df['voltage'] - train_mean) / train_std

X = create_sequences(df['voltage_scaled'].values, time_steps=30)
# Reshape for LSTM [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1) 

# 3. Build the LSTM Autoencoder Architecture
model = Sequential([
    # Encoder
    LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    # The latent representation (bottleneck)
    RepeatVector(X.shape[1]),
    # Decoder
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 4. Train the Model (In practice, train ONLY on nominal/normal data)
history = model.fit(X, X, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# 5. Predict and Calculate Reconstruction Error
X_pred = model.predict(X)
train_mae_loss = np.mean(np.abs(X_pred - X), axis=1)

# 6. Apply IQR Boundary Rule for the Anomaly Threshold
Q1 = np.percentile(train_mae_loss, 25)
Q3 = np.percentile(train_mae_loss, 75)
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR

# 7. Flag Anomalies
anomalies = train_mae_loss > threshold
print(f"Number of anomalies detected: {np.sum(anomalies)}")

# 8. Visualization
plt.figure(figsize=(14, 5))
plt.plot(df.index[30-1:], train_mae_loss, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='IQR Threshold')
plt.title('Spacecraft Telemetry: Anomaly Detection via Reconstruction Error')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import ast
import os

def load_telemetry_channel(channel_id, data_dir='data'):
    """
    Loads the training data, test data, and anomaly labels for a specific NASA sensor.
    """
    # 1. Load the raw NumPy arrays
    train_path = os.path.join(data_dir, 'train', f'{channel_id}.npy')
    test_path = os.path.join(data_dir, 'test', f'{channel_id}.npy')
    
    # NASA data is multidimensional; we typically target the first column (the main sensor value)
    X_train = np.load(train_path)[:, 0] 
    X_test = np.load(test_path)[:, 0]
    
    # 2. Extract the true anomaly indices from the master CSV
    labels_df = pd.read_csv(os.path.join(data_dir, 'labeled_anomalies.csv'))
    
    # Isolate the row for our specific channel
    channel_data = labels_df[labels_df['chan_id'] == channel_id].iloc[0]
    
    # NASA stored the anomaly sequences as a string representation of a list (e.g., "[[2400, 2800]]")
    # ast.literal_eval safely parses that string back into a real Python list
    anomaly_sequences = ast.literal_eval(channel_data['anomaly_sequences'])
    
    # 3. Create a binary label array for the test set (0 = Normal, 1 = Anomaly)
    y_test = np.zeros(len(X_test))
    for sequence in anomaly_sequences:
        start_idx, end_idx = sequence
        y_test[start_idx:end_idx] = 1.0
        
    print(f"Loaded Channel {channel_id} ({channel_data['spacecraft']})")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Anomalous timesteps: {int(np.sum(y_test))}")
    
    return X_train, X_test, y_test

# --- Test the Loader ---
# 'T-1' is a temperature sensor channel from the dataset
train_data, test_data, test_labels = load_telemetry_channel('T-1')

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# 1. Statistical Scaling (Fit ONLY on training data to prevent data leakage)
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

# 2. Sliding Window Generator
def create_sequences(X, time_steps=50):
    """
    Slices the continuous telemetry into overlapping windows of 'time_steps' length.
    """
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : (i + time_steps)])
    return np.array(Xs)

TIME_STEPS = 50

X_train = create_sequences(train_data_scaled, TIME_STEPS)
X_test = create_sequences(test_data_scaled, TIME_STEPS)

# Keras LSTMs require a 3D tensor shape: [samples, time_steps, features]
# Our feature count is 1 (just the sensor reading)
print(f"Training Tensor Shape: {X_train.shape}")

# 3. Define the LSTM Autoencoder Architecture
model = Sequential([
    # Encoder: Compresses the 50-timestep window into a 32-dimensional latent vector
    LSTM(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    
    # Bottleneck: Duplicates the latent vector so the decoder has input for every timestep
    RepeatVector(X_train.shape[1]),
    
    # Decoder: Reconstructs the 50-timestep window
    LSTM(32, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_train.shape[2]))
])

model.compile(optimizer='adam', loss='mse')

# 4. Train the Model
# We train X_train to predict X_train (unsupervised reconstruction)
print("\nCommencing Model Training...")
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
    shuffle=False 
)

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# 1. Establish the Anomaly Threshold from Normal Data
# We predict on the training set to see what the "normal" error looks like
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=(1, 2))

# Apply the IQR Boundary Rule
Q1 = np.percentile(train_mae_loss, 25)
Q3 = np.percentile(train_mae_loss, 75)
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR
print(f"Dynamic Anomaly Threshold calculated at: {threshold:.4f}\n")

# 2. Predict on the Unseen Test Data
print("Scanning Test Telemetry for Hardware Failures...")
X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=(1, 2))

# 3. Flag the Anomalies
# If the error is strictly greater than our threshold, flag it as a failure
predicted_anomalies = test_mae_loss > threshold

# 4. Align the Sliding Window Offset
# Because our sliding window consumes the first 50 timesteps, our prediction array
# is 50 timesteps shorter than the original raw test_data and test_labels.
# We pad the beginning of the predictions with 'False' to align them for scoring.
aligned_predictions = np.pad(predicted_anomalies, (TIME_STEPS, 0), mode='constant', constant_values=False)

# 5. Evaluate against NASA's True Labels
print("\n--- NASA Model Performance Report ---")
print(classification_report(test_labels, aligned_predictions, target_names=['Normal (0)', 'Anomaly (1)']))

# 6. Visualize the Hardware Failure Detection
plt.figure(figsize=(14, 6))

# Plot the raw sensor data
plt.plot(test_data, label='Raw Sensor Telemetry', color='steelblue', alpha=0.7)

# Overlay the True NASA Anomalies in Yellow
plt.fill_between(range(len(test_data)), test_data.min(), test_data.max(), 
                 where=(test_labels == 1), color='gold', alpha=0.3, label='True NASA Anomaly')

# Overlay Our Model's Predictions in Red
anomalous_indices = np.where(aligned_predictions == True)[0]
plt.scatter(anomalous_indices, test_data[anomalous_indices], color='red', s=10, label='LSTM Predicted Failure')

plt.title(f"Spacecraft Telemetry Anomaly Detection (Sensor T-1)")
plt.xlabel("Time Steps")
plt.ylabel("Sensor Value")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
