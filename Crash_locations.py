"""Plot points overlayed on a map of crash locations in Allegheny County 2021, saved to an HTML file."""

import pandas as pd
import folium
import re

# Read the CSV file
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

# Convert latitude and longitude to degrees and create new columns
df['LAT_DEG'] = df['LATITUDE'].astype(str).apply(dms_to_degrees)
df['LON_DEG'] = -1 * df['LONGITUDE'].astype(str).apply(dms_to_degrees)

# Remove any rows where the conversion function returned None
df = df.dropna(subset=['LAT_DEG', 'LON_DEG'])
print("Dropped rows:", len(df) - len(df.dropna(subset=['LAT_DEG', 'LON_DEG'])))

# Create a map centered on Pittsburgh, PA
map = folium.Map(location=[40.4406, -79.9959], zoom_start=10)

# Add a marker for each crash
for index, row in df.iterrows():
    folium.Marker([row['LAT_DEG'], row['LON_DEG']]).add_to(map)

# Display the map
map.save("crash_map.html")
