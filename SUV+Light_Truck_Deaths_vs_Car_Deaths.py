import sys
import psycopg2
import pandas as pd

# Connect to PostgreSQL server
# Insert database name, username, password, server address, and port here.
# Leave blank if empty
DB_NAME = ('postgres')
DB_USER = ('postgres')
DB_PASS = ('')
DB_HOST = ('localhost')
DB_PORT = ('5432')

try:
    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT)
    print("Database connected successfully!")
except:
    print("Database failed to connect.")
    sys.exit()

# Create the necessary tables in PostgreSQL
cur = conn.cursor()
try:
    cur.execute('''
        CREATE TABLE vehicles (
            CRN TEXT,
            UNIT_NUM INTEGER,
            MAKE_CD TEXT,
            MODEL_YR INTEGER,
            MODEL_CATEGORY TEXT,
            DAMAGE_IND TEXT,
            PRIMARY KEY (CRN, UNIT_NUM)
        );
    ''')
except psycopg2.errors.DuplicateTable:
    # State if table already exists
    print("Table vehicles already exists.")
conn.commit()
try:
    cur.execute('''
        CREATE TABLE fatalities (
            CRN TEXT PRIMARY KEY,
            FATAL_COUNT INTEGER,
            INJURY_COUNT INTEGER,
            PERSON_COUNT INTEGER
        );
    ''')
except psycopg2.errors.DuplicateTable:
    # State if table already exists
    print("Table fatalities already exists.")
conn.commit()

# Load VEHICLE data into a pandas dataframe
df = pd.read_csv('VEHICLE_ALLEGHENY_2021.csv')
# Drop rows with missing data
df = df.dropna(subset=['CRN', 'UNIT_NUM', 'MAKE_CD', 'MODEL_YR', 'BODY_TYPE', 'DAMAGE_IND'])

# Comment in for vehicles Model Year 2015 and up, the year SUVs began outselling sedans
# Rules out old sedans that may be skewing results with higher fatality rate due to fewer safety features
# df = df[df['MODEL_YR'] >= 2015]

# Categorize the body types into cars, trucks/suvs, motorcycles, and vans
df.loc[df['BODY_TYPE'].isin([1,2,3,4,5,6]), 'MODEL_CATEGORY'] = 'Cars'
df.loc[df['BODY_TYPE'].isin([10,11,12,50,51,69,72,73]), 'MODEL_CATEGORY'] = 'Light Trucks + SUVs'
df.loc[df['BODY_TYPE'].isin([20,23,24,25,28,29]), 'MODEL_CATEGORY'] = 'Motorcycles'
df.loc[df['BODY_TYPE'].isin([39,40,41,42,49]), 'MODEL_CATEGORY'] = 'Vans'

# Insert data from pandas dataframe into PostgreSQL table
for index, row in df.iterrows():
    cur.execute('''
        INSERT INTO vehicles (
            CRN, 
            UNIT_NUM, 
            MAKE_CD, 
            MODEL_YR, 
            MODEL_CATEGORY, 
            DAMAGE_IND
        )
        VALUES (
            %s, 
            %s, 
            %s, 
            %s, 
            %s, 
            %s
        );
    ''',
        (int(row['CRN']),
        int(row['UNIT_NUM']),
        row['MAKE_CD'],
        int(row['MODEL_YR']),
        row['MODEL_CATEGORY'],
        int(row['DAMAGE_IND']))
    )

# Load CRASH data into a pandas dataframe
df = pd.read_csv('CRASH_ALLEGHENY_2021.csv')
# Drop rows with missing data
df = df.dropna(subset=['CRN', 'FATAL_COUNT', 'INJURY_COUNT', 'PERSON_COUNT'])
# Insert data from pandas dataframe into PostgreSQL table
for index, row in df.iterrows():
    cur.execute('''
        INSERT INTO fatalities (
            CRN, 
            FATAL_COUNT, 
            INJURY_COUNT, 
            PERSON_COUNT
        )
        VALUES (
            %s, 
            %s, 
            %s, 
            %s
        );
    ''',
        (int(row['CRN']),
        int(row['FATAL_COUNT']),
        int(row['INJURY_COUNT']),
        int(row['PERSON_COUNT']))
    )

conn.commit()

# Perform a join operation on the two tables using the CRN column
cur.execute('''
    SELECT vehicles.MODEL_CATEGORY, 
    COUNT(
        CASE WHEN fatalities.FATAL_COUNT > 0 
        THEN 1 END) 
        AS FATAL_CRASHES, 
    COUNT(
        CASE WHEN fatalities.FATAL_COUNT = 0 
        THEN 1 END) 
        AS NONFATAL_CRASHES
    FROM vehicles
    JOIN fatalities ON vehicles.CRN = fatalities.CRN
    GROUP BY vehicles.MODEL_CATEGORY
    ORDER BY vehicles.MODEL_CATEGORY;
''')

# Get the results and put them into a Pandas DataFrame
results = pd.DataFrame(cur.fetchall(), columns=['MODEL_CATEGORY', 'FATAL_CRASHES', 'NONFATAL_CRASHES'])

# Calculate the total number of crashes per category
results['TOTAL_CRASHES'] = results['FATAL_CRASHES'] + results['NONFATAL_CRASHES']

# Calculate the percentage of fatal and nonfatal crashes per category
results['FATAL_CRASHES_PERCENTAGE'] = results['FATAL_CRASHES'] / results['TOTAL_CRASHES'] * 100
results['NONFATAL_CRASHES_PERCENTAGE'] = results['NONFATAL_CRASHES'] / results['TOTAL_CRASHES'] * 100

# Print the results
print(results)

# Close the cursor and connection
cur.close()
conn.close()