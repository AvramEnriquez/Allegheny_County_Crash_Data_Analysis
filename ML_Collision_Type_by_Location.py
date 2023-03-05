"""Attempt to calculate collision type given a latitude and longitude input for Allegheny County.

COLLISION TYPE
0 - Non-collision
1 - Rear-end
2 - Head-on
3 - Backing
4 - Angle
5 - Sideswipe (same dir.)
6 - Sideswipe (Opposite dir.)
7 - Hit fixed object
8 - Hit pedestrian
9 - Other/Unknown (Expired)

"""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

import tensorflow as tf
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data and convert latitude and longitude to degrees
df = pd.read_csv('CRASH_ALLEGHENY_2021.csv')

# Replace values > 1 in 'FATAL_COUNT' column with 1
df.loc[df['COLLISION_TYPE'] > 9, 'COLLISION_TYPE'] = 9

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
y = df['COLLISION_TYPE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # output layer has 10 neurons for the 10 possible collision types
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Test the model with a single set of latitude and longitude
test_data = [[40.44326, -79.98994]] # adjust the values here to test different locations
scaled_test_data = scaler.transform(test_data)
prediction = model.predict(scaled_test_data)[0]
predicted_class = np.argmax(prediction)

print(f"The highest likelihood collision type for location: {test_data[0][0]}, {test_data[0][1]} is {predicted_class}.")