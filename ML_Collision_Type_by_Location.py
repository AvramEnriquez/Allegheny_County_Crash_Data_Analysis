"""Attempt to calculate likelihood of collision types given a latitude and longitude input for Allegheny County. Runs a grid search via GridSearchCV to loop 
through all possible hyperparameters to achieve the highest accuracy. 

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
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# Load data and convert latitude and longitude to degrees
df = pd.read_csv('CRASH_ALLEGHENY_2021.csv')

# Replace values > 9 in 'COLLISION_TYPE' column with 9
df.loc[df['COLLISION_TYPE'] > 9, 'COLLISION_TYPE'] = 9

# Define the hyperparameters to search over
param_grid = {'batch_size': [16, 32, 64],
              'epochs': [50, 100, 150],
              'num_hidden_layers': [1, 2, 3],
              'num_neurons': [8, 16, 32],
              'learning_rate': [0.001, 0.01, 0.1]}

# Define a function to create the model with the specified hyperparameters, defaults to given otherwise
# Fitted with early stopping
def create_model(learning_rate=0.001, num_hidden_layers=3, num_neurons=32, epochs=150, batch_size=16):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_neurons, activation='relu', input_shape=(2,)))
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stop])
    return model

# Define a function to convert latitude and longitude in DMS format to float
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

# Define early stopping callback
# Adjust patience variable to specify the number of epochs with no improvement until it stops
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

"""Comment out from here-"""
# # Create the grid search object
# model = KerasClassifier(model=create_model, learning_rate=0.001, num_hidden_layers=2, num_neurons=32, verbose=1)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# # Fit the grid search to the data
# grid_result = grid.fit(X_train, y_train, verbose=1)

# # Print the best hyperparameters and the corresponding validation accuracy
# print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
# # 
"""-To here to remove grid search. Adjust hyperparameters in create_model() function for manual model creation. Running takes a couple of hours.""" 

"""
From my results:
Best: 0.33998489720574393 using {'batch_size': 16, 'epochs': 150, 'learning_rate': 0.001, 'num_hidden_layers': 3, 'num_neurons': 32}
"""

"""Comment out first model variable if manually providing hyperparameters"""
# model = create_model(**grid_result.best_params_)
model = create_model()

# Test the model with a single set of latitude and longitude
test_data = [[40.454009, -79.912390]] # adjust the values here to test different locations
scaled_test_data = scaler.transform(test_data)
predicted_probs = model.predict(scaled_test_data)[0]
predicted_classes = np.argsort(predicted_probs)[::-1]

# Print % likelihood for all collision types
print(f"The likelihood percentages of collision types for location {test_data[0][0]}, {test_data[0][1]} are:")
for i in range(10):
    print(f"Collision Type {predicted_classes[i]}: {predicted_probs[predicted_classes[i]]*100:.2f}%")
