#This script should fill out excel table with the time of arriving and leaving of employees in the office
#With the help of YOLOv8 model we will detect human and save the time of arrival and leave
#In this version I will test with hours instead of days, and days will be instead of months, in this way it is easier to test the work of this system, but the logic is the same with days and months

import cv2
from IPython.display import display
from ultralytics import YOLO
import pandas as pd
import datetime as dt
import numpy as np
import os

firstseen = [0,0]
lastseen = [0,0]

seenattime = [0,0]
atwork = [0,0] #ad rus

old_time = dt.datetime.today()
actual_h = old_time.hour

worktime = np.array([[actual_h, firstseen[0], lastseen[0], firstseen[1], lastseen[1]]])
# worktime = np.append(worktime, [[actual_h, firstseen[0], lastseen[0], firstseen[1], lastseen[1]]], axis=0)

# Load the YOLOv8 model
model = YOLO('runs\content\\runs\detect\\train\weights\\best.pt')

# Open camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

def write_firstdata_tofile():
    """Function to save the data of arrival
      to a csv file"""

    if os.path.exists(f'tables\office_{exact_time.month}_{exact_time.day}_{exact_time.year}.csv'): #if we have already that file...
        df = pd.read_csv(f'tables\office_{exact_time.month}_{exact_time.day}_{exact_time.year}.csv', index_col= 0) #we will read it and ...
        if actual_h not in df.loc[:, 'time'].values:
            df.loc[len(df)] = [actual_h, firstseen[0], lastseen[0], firstseen[1], lastseen[1]] #add new row to df.
        else:
            if df.loc[len(df)-1, 'Ad_f' ] == 0:  #check not to overwrite already registrated data
                df.loc[len(df)-1, 'Ad_f' ] = firstseen[0]
            if df.loc[len(df)-1, 'Rus_f' ] == 0:
                df.loc[len(df)-1, 'Rus_f' ] = firstseen[1]
    else:
        df = pd.DataFrame([[actual_h, firstseen[0], lastseen[0], firstseen[1], lastseen[1]]], columns=['time', 'Ad_f', 'Ad_l', 'Rus_f', 'Rus_l']) #create the df

    # save the dataframe as a csv file
    df.to_csv(f'tables\office_{exact_time.month}_{exact_time.day}_{exact_time.year}.csv')
    display(df)

def write_lastdata_tofile():
    """Function to save the data of leave 
      to a csv file"""

    if os.path.exists(f'tables\office_{exact_time.month}_{exact_time.day}_{exact_time.year}.csv'): #if we have already that file...
        df = pd.read_csv(f'tables\office_{exact_time.month}_{exact_time.day}_{exact_time.year}.csv', index_col= 0) #we will read it and ...
        df.loc[len(df)-1, 'Ad_l' ] = lastseen[0]
        df.loc[len(df)-1, 'Rus_l' ] = lastseen[1]
    else:
        df = pd.DataFrame([[actual_h - 1, firstseen[0], lastseen[0], firstseen[1], lastseen[1]]], columns=['time', 'Ad_f', 'Ad_l', 'Rus_f', 'Rus_l']) #create the df

    # save the dataframe as a csv file
    df.to_csv(f'tables\office_{exact_time.month}_{exact_time.day}_{exact_time.year}.csv')
    display(df)

def seen_times(results):
    '''Function to get the time,
      when the employee is first seen'''

    for result in results:
            boxes = result.boxes
            for box in boxes:
                #print(f'{box.cls} : {box.conf * 100}%')
                if box.conf * 100 > 80: #setting up the confidence level of detecting the person
                    if box.cls == 1:
                        seenattime[1] = float(f'{exact_time.hour}.{exact_time.minute}')
                        i = 1
                        if atwork[i] == 0:
                            firstseen[i] = seenattime[i]
                            print(f'first seen {i} at: ', seenattime[i])
                            atwork[i] = 1
                            write_firstdata_tofile()
                            
                    elif box.cls == 0:
                        seenattime[0] = float(f'{exact_time.hour}.{exact_time.minute}')
                        i = 0
                        if atwork[i] == 0:
                            firstseen[i] = seenattime[i]
                            print(f'first seen {i} at: ', seenattime[i])
                            atwork[i] = 1
                            write_firstdata_tofile()

def lastseen_times():
    '''Function to get the time,
      when the employee is last seen'''
    global actual_h
    global worktime
    global firstseen
    global lastseen
    if actual_h != exact_time.hour :
        actual_h = exact_time.hour

        for i in range(len(atwork)):
            if atwork[i] == 1:
                print(f'last seen {i} at: ', seenattime[i])
                lastseen[i] = seenattime[i]
                atwork[i] = 0

        
        write_lastdata_tofile()
        firstseen = [0,0]
        lastseen = [0,0]


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    exact_time = dt.datetime.today() #after every frame get the exact time

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame,  verbose = False)

        seen_times(results)

        lastseen_times()
                
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()