"""Attempt to calculate percent chance of getting into a fatal car crash given a latitude and longitude input for Allegheny County."""

import tensorflow as tf
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data and convert latitude and longitude to degrees
df = pd.read_csv('CRASH_ALLEGHENY_2021.csv')

def dms_to_degrees(dms_str):
    parts = re.split(':|\s+', dms_str)
    if len(parts) != 3:
        return None
    try:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    total_degrees = degrees + minutes/60 + seconds/3600
    return total_degrees

df['LATITUDE'] = df['LATITUDE'].astype(str).apply(dms_to_degrees)
df['LONGITUDE'] = -1 * df['LONGITUDE'].astype(str).apply(dms_to_degrees)
df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

# Split data into training and testing sets
X = df[['LATITUDE', 'LONGITUDE']]
y = df['FATAL_COUNT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=75, batch_size=64, validation_data=(X_test, y_test))

# Test the model with a single set of latitude and longitude
test_data = [[40.439188, -80.010702]] # adjust the values here to test different locations
scaled_test_data = scaler.transform(test_data)
prediction = model.predict(scaled_test_data)[0][0] * 100
print(f"The percentage chance of getting into a fatal crash at {test_data[0][0]}, {test_data[0][1]} is {prediction:.2f}%")
