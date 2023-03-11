# Allegheny_County_Crash_Data_Analysis
Data analysis of Allegheny County crash stats for 2021.
Made with use of ChatGPT. Validated, fixed, and cleaned up by me.

**- Crash_Locations.py:**

This code reads in a CSV file containing data on crashes that occurred in Allegheny County in 2021. It then converts the latitude and longitude coordinates from degrees, minutes, and seconds format to decimal degrees format and adds new columns to the DataFrame containing the converted values.

After dropping any rows where the conversion function returned None, the code creates a folium map centered on Pittsburgh, PA with a zoom level of 10. For each row in the DataFrame, a marker is added to the map with the latitude and longitude coordinates corresponding to that row.

Finally, the map is saved as an HTML file named "crash_map.html" in the same directory as the script. This HTML file can be opened in a web browser to view the map with the crash locations.


**- Crashes_by_Hr_of_Day.py:**

Plot crashes on a line graph based off time of day. The plot shows a trend of higher number of crashes during late afternoon to evening hours, with a peak around 4 PM.

![Figure_1](https://user-images.githubusercontent.com/120682270/223276823-833e9758-63f5-4eda-add8-8617343285f4.png)


**- ML_Collision_Type_by_Location.py:**

This is a machine learning code that attempts to predict the likelihood of different types of collisions for a specific location in Allegheny County, based on latitude and longitude data, based off historical data from 2017-2021.

First, the code loads the data from a CSV file into a pandas DataFrame, replaces values greater than 9 in the 'COLLISION_TYPE' column with 9, and converts the latitude and longitude data from DMS format to degrees. Then it splits the data into training and testing sets, standardizes the features using StandardScaler, and defines an early stopping callback.

The code defines two functions: 'create_model()' and 'dms_to_degrees()'. 'dms_to_degrees()' converts latitude and longitude data in DMS format to floating-point degrees. 'create_model()' creates a deep learning model using TensorFlow and Keras, with a variable number of hidden layers, neurons, and epochs, and returns the trained model.

Next, the code defines a dictionary of hyperparameters to search over, and uses GridSearchCV to perform a grid search for the optimal hyperparameters. However, the grid search has been commented out and the best model found by the grid search is printed and specified manually in the 'create_model()' function, as the grid search takes a couple of hours to run.

Finally, the code tests the model with a single set of latitude and longitude data, and prints the likelihood percentages for all collision types. The likelihoods are calculated using the 'predict()' method of the trained model, and the results are sorted in descending order of likelihood.

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

![Screenshot 2023-03-06 at 6 02 44 PM](https://user-images.githubusercontent.com/120682270/223276894-e4c0e7de-5a2d-466d-9b35-b4158c9b5108.png)


**- ML_Fatal_Crash_%\_by_Location.py:**

Attempt to calculate percent chance of getting into a fatal car crash given a latitude and longitude input for Allegheny County, based off historical data from 2017-2021. The latitude and longitude data, along with the target variable (FATAL_COUNT), are split into training and testing sets, and the features are standardized using the StandardScaler.

Next, a neural network model is defined using the Sequential class from the tensorflow.keras library. The model consists of three fully connected layers, with the first layer having 32 neurons, the second layer having 16 neurons, and the output layer having a single neuron with a sigmoid activation function. The model is compiled with binary cross-entropy loss and the Adam optimizer, and accuracy is used as a metric.

The model is trained using the fit method with the training data, and the accuracy is measured on the validation data. After training, the model is used to predict the percentage chance of getting into a fatal car crash at a single set of latitude and longitude values.

Finally, the model is tested to find the highest percentage prediction of fatal crashes by latitude and longitude using a for loop to iterate over a range of latitude and longitude values. The latitude and longitude values are scaled to decimal degrees, and the prediction is made using the model's predict method. The latitude and longitude values with the highest prediction are printed. However, this part of the code is commented out as it takes a long time to run due to the for loop, so it is not recommended to run it without adjusting the range and step size.

![Screenshot 2023-03-06 at 6 07 50 PM](https://user-images.githubusercontent.com/120682270/223277119-318f8cc7-6481-462d-a707-178f762c7a43.png)


**- SUV+Light_Truck_Deaths_vs_Car_Deaths.py:**

This code is used to connect to a PostgreSQL server, create two tables, load data from two CSV files into the tables, and then perform a join operation on the tables to calculate pedestrian crash percentage and fatality rate per category of vehicles involved in crashes, using data from 2017-2021..

The code sets variables for the database name, username, password, server address, and port. These variables are used to connect to a PostgreSQL server using psycopg2. The code then attempts to connect to the server using a try-except block. If the connection is successful, the code prints a message indicating that the connection was successful. If the connection fails, the code prints a message indicating that the connection failed and exits the program.

Next, the code creates two tables in the PostgreSQL server using cur.execute(). The first table is called 'vehicles' and has columns for the crash report number (CRN), unit number, make code, model year, and model category. The CRN alone cannot be the primary key for this table--reason being that multiple vehicles can be in a single crash (1 CRN) and so the CRN and unit number (Assigned number to vehicles in a crash) together form the primary key for this table. The second table is called 'fatalities' and has columns for the CRN, pedestrian count, pedestrian death count. The CRN is unique here and is the primary key for this table.

After creating the tables, the code loads data in from VEHICLE_ALLEGHENY_2017-2021.csv and CRASH_ALLEGHENY_2017-2021.csv. 
The VEHICLE_ALLEGHENY CSV files contain data on vehicle types and models involved in crashes in Allegheny County from 2017-2021. The code loads this data into the vehicles table. It drops any rows with missing data and categorizes the vehicles into one of six categories: cars, light trucks/suvs, heavy trucks, motorcycles, buses, and vans. Anything not categorized will be categorized as NaN. The CRASH_ALLEGHENY CSV files contain data on fatalities of crashes in Allegheny County from 2017-2021. The code loads this data into the fatalities table. It drops any rows with missing data.

There is a line that can be un-commented to only include vehicles with model year 2015 or newer--the year SUVs and light trucks began outselling cars. This could potentially rule out older vehicles (when smaller cars were more popular) that may be skewing results due to fewer active safety features.

After loading the data, the code performs a join operation on the two tables using the CRN column. It calculates the pedestrian crash percentage and fatality rate per category of vehicle using pd.DataFrame().

Finally, the code prints the results and closes the cursor and connection to the PostgreSQL server.

All model years:

![Screenshot 2023-03-09 at 8 46 41 PM](https://user-images.githubusercontent.com/120682270/224202413-64a06d7d-d355-4760-a33e-cc13a9096357.png)

Model years 2015 and up:

![Screenshot 2023-03-09 at 8 47 50 PM](https://user-images.githubusercontent.com/120682270/224202434-a89e6b5e-879b-4092-939c-344426d7af0b.png)


**- Young_Mid_Old_Fatal_Crash_%.py:**

Calculate the percentages of young, old, and middle aged fatal crashes versus total crashes. It calculates the total number of fatal crashes for each age group (young_fatal, old_fatal, and middle_fatal) by selecting the rows in the DataFrame where the driver count for that age group is greater than 0.

![Screenshot 2023-03-06 at 6 10 30 PM](https://user-images.githubusercontent.com/120682270/223277539-d0e4e7ad-2fb2-43e1-b5f1-9f8339303f63.png)
