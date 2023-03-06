"""Calculate and plot the percentages of young, old, and middle aged fatal crashes versus total crashes."""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('CRASH_ALLEGHENY_2021.csv')

# Combine columns DRIVER_COUNT_16YR to DRIVER_COUNT_20YR into one column called DRIVER_COUNT_YOUNG
df['DRIVER_COUNT_YOUNG'] = df['DRIVER_COUNT_16YR'] + df['DRIVER_COUNT_17YR'] + df['DRIVER_COUNT_18YR'] + df['DRIVER_COUNT_19YR'] + df['DRIVER_COUNT_20YR']

# Combine columns DRIVER_COUNT_50_64YR and DRIVER_COUNT_65_74YR into one column called DRIVER_COUNT_OLD
df['DRIVER_COUNT_OLD'] = df['DRIVER_COUNT_50_64YR'] + df['DRIVER_COUNT_65_74YR']

# Calculate the total crashes for each category
young_crashes = df['DRIVER_COUNT_YOUNG'].sum()
old_crashes = df['DRIVER_COUNT_OLD'].sum()
middle_crashes = df['PERSON_COUNT'].sum() - young_crashes - old_crashes

# Calculate the total fatal crashes for each age group
young_fatal = df.loc[df['DRIVER_COUNT_YOUNG'] > 0, 'FATAL_COUNT'].sum()
old_fatal = df.loc[df['DRIVER_COUNT_OLD'] > 0, 'FATAL_COUNT'].sum()
middle_fatal = df.loc[(df['DRIVER_COUNT_YOUNG'] == 0) & (df['DRIVER_COUNT_OLD'] == 0), 'FATAL_COUNT'].sum()

print(f'Young fatal crash percentage: {round((young_fatal/young_crashes)*100,2)}%')
print(f'Middle-aged fatal crash percentage: {round((middle_fatal/middle_crashes)*100,2)}%')
print(f'Old fatal crash percentage: {round((old_fatal/old_crashes)*100,2)}%')
