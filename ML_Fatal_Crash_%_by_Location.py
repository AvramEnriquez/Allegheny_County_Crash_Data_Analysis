"""Attempt to calculate percent chance of getting into a fatal car crash given a latitude and longitude input for Allegheny County."""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

import tensorflow as tf
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from 2017-2021
filenames = ['CRASH_ALLEGHENY_2017.csv', 'CRASH_ALLEGHENY_2018.csv', 'CRASH_ALLEGHENY_2019.csv', 'CRASH_ALLEGHENY_2020.csv', 'CRASH_ALLEGHENY_2021.csv']
dfs = []
for filename in filenames:
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

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
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Test the model with a single set of latitude and longitude
test_data = [[40.434043, -79.992462]] # adjust the values here to test different locations
scaled_test_data = scaler.transform(test_data)
prediction = model.predict(scaled_test_data)[0][0] * 100
print(f"The percentage chance of getting into a fatal crash at {test_data[0][0]}, {test_data[0][1]} is {prediction:.2f}%")

# # Test the model to find the highest % fatality by latitude and longitude, scaled to 1 decimal point
# max_prediction = 0
# max_lat = None
# max_lon = None
# for lat in range(40000000, 42000000, 100000): # To increase decimal and exponentially increase time the for loop runs, remove zeros on the step size (100000)
#     for lon in range(-81000000, -79000000, 100000): # To increase decimal and exponentially increase time the for loop runs, remove zeros on the step size (100000)
#         test_data = [[lat / 1000000, lon / 1000000]]
#         scaled_test_data = scaler.transform(test_data)
#         prediction = model.predict(scaled_test_data)[0][0] * 100
#         if prediction > max_prediction:
#             max_prediction = prediction
#             max_lat = lat / 1000000
#             max_lon = lon / 1000000
# print(f"The highest prediction of {max_prediction:.2f}% is for latitude {max_lat:.6f} and longitude {max_lon:.6f}.")

