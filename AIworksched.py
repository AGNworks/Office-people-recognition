#This script should fill out excel table with the time of arriving and leaving of employees in the office
#With the help of YOLOv8 model we will detect human and save the time of arrival and leave

import cv2
from IPython.display import display
from ultralytics import YOLO
import pandas as pd
import datetime as dt
import numpy as np

firstseen = [0,0]
lastseen = [0,0]

seenattime = [0,0]
atwork = [0,0] #ad rus

today_date = dt.datetime.today()
actual_h = today_date.hour

worktime = np.array([[actual_h, firstseen[0], lastseen[0], firstseen[1], lastseen[1]]])

# worktime = np.append(worktime, [[actual_h, firstseen[0], lastseen[0], firstseen[1], lastseen[1]]], axis=0)

def write_data_tofile(worktime):
    """Function to save the data to a csv file"""
    DF = pd.DataFrame(worktime, columns=['time', 'Ad_f', 'Ad_l', 'Rus_f', 'Rus_l'])
 
    # save the dataframe as a csv file
    DF.to_csv(f'office_{today_date.month}_{today_date.day}_{today_date.year}.csv')
    display(DF)


# Load the YOLOv8 model
model = YOLO('runs\content\\runs\detect\\train\weights\\best.pt')

# Open camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame,  verbose = False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                #print(f'{box.cls} : {box.conf * 100}%')
                if box.conf * 100 > 50:
                    if box.cls == 1:
                        seenattime[1] = float(f'{today_date.hour}.{today_date.minute}')
                        i = 1
                        if atwork[i] == 0:
                            firstseen[i] = seenattime[i]
                            print(f'first seen {i} at: ', seenattime[i])
                            atwork[i] = 1
                            
                    elif box.cls == 0:
                        seenattime[0] = float(f'{today_date.hour}.{today_date.minute}')
                        i = 0
                        if atwork[i] == 0:
                            firstseen[i] = seenattime[i]
                            print(f'first seen {i} at: ', seenattime[i])
                            atwork[i] = 1

        
                
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

for i in range(len(atwork)):
    if atwork[i] == 1:
        print(f'last seen {i} at: ', seenattime[i])
        lastseen[i] = seenattime[i]

worktime = np.append(worktime, [[actual_h, firstseen[0], lastseen[0], firstseen[1], lastseen[1]]], axis=0)
write_data_tofile(worktime)


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()