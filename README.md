# Allegheny_County_Crash_Data_Analysis
Data analysis of Allegheny County crash stats for 2021.
Made with help from ChatGPT for learning purposes. Validated, fixed, and cleaned up by me.

- Crash_Locations.py:
Creates an html file called "crash_map.html" which plots all crashes in Allegheny County 2021 on a map.


- Crashes_by_Hr_of_Day.py:
Plot crashes on a line graph based off time of day.

![Figure_1](https://user-images.githubusercontent.com/120682270/223276823-833e9758-63f5-4eda-add8-8617343285f4.png)


- ML_Collision_Type_by_Location.py:
Attempt to calculate likelihood of collision types given a latitude and longitude input for Allegheny County. Runs a grid search via GridSearchCV to loop 
through all possible hyperparameters to achieve the highest accuracy. Part of the code can be commented out to manually provide hyperparameters.

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


- ML_Fatal_Crash_%_by_Location.py:
Attempt to calculate percent chance of getting into a fatal car crash given a latitude and longitude input for Allegheny County.

![Screenshot 2023-03-06 at 6 07 50 PM](https://user-images.githubusercontent.com/120682270/223277119-318f8cc7-6481-462d-a707-178f762c7a43.png)


- Young_Mid_Old_Fatal_Crash_%.py:
Calculate and plot the percentages of young, old, and middle aged fatal crashes versus total crashes.

![Screenshot 2023-03-06 at 6 10 30 PM](https://user-images.githubusercontent.com/120682270/223277539-d0e4e7ad-2fb2-43e1-b5f1-9f8339303f63.png)
