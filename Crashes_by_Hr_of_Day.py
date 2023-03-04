"""Plot crashes based off time of day."""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('CRASH_ALLEGHENY_2021.csv')

# Filter out values above 23 in HOUR_OF_DAY column
data = data[data['HOUR_OF_DAY'] <= 23]

# Group the crashes by hour of the day and count the number of occurrences
crashes_by_hour = data.groupby('HOUR_OF_DAY').size()

# Plot the data
plt.plot(crashes_by_hour.index, crashes_by_hour.values)

# Customize the plot
plt.title('Number of Crashes by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Crashes')
plt.xticks(range(24), range(24))  # Set the x-axis tick locations and labels
plt.grid(True)

# Show the plot
plt.show()
