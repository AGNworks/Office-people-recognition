#This script should fill out excel table with the time of arriving and leaving of employees in the office
#With the help of YOLOv8 model we will detect human and save the time of arrival and leave

import cv2
from ultralytics import YOLO
import pyttsx3
import pandas as pd
import datetime as dt

Rustam_fl = [0,0]
Adam_fl = [0,0]

df = pd.DataFrame(
    {
        'Hour' : [1], #for tests I will use hours instead of days 
        'Rus_f' : [2], #Rustam first seen at 12:00 for example
        'Rus_l' : [3], #Rustam last seen at 13:00 for example
        'Adam_f' : [2],
        'Adam_l' : [44], 
    }
)

today_date = dt.datetime.today()
actual_h = today_date.hour

df.Hour[0] = actual_h
df.Rus_f[0] = Rustam_fl[0]
df.Rus_l[0] = Rustam_fl[1]
df.Adam_f[0] = Adam_fl[0]
df.Adam_l[0] = Adam_fl[1]

# Setting up the Excel writer
writer = pd.ExcelWriter(f"office_{today_date.month}_{today_date.day}_{today_date.year}.xlsx", engine="xlsxwriter")
df.to_excel(writer, sheet_name="Sheet1", startrow=1, header=False, index=False)

workbook = writer.book
worksheet = writer.sheets["Sheet1"]

# Get the dimensions of the dataframe.
(max_row, max_col) = df.shape

# Create a list of column headers, to use in add_table().
column_settings = [{"header": column} for column in df.columns]

# Add the Excel table structure. Pandas will add the data.
worksheet.add_table(0, 0, max_row, max_col - 1, {"columns": column_settings})

# Make the columns wider for clarity.
worksheet.set_column(0, max_col - 1, 12)

# Close the Pandas Excel writer and output the Excel file.
writer.close()

""" #Speaking engine for further functions
engine = pyttsx3.init()
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')

def speaking(text):
    engine.say(text)   #say something
    engine.runAndWait() #run the engine and wait for it to finish """
    

""" # Load the YOLOv8 model
model = YOLO('D:\SAJAT\\00_ML\MY PROJECTS\People recognization\\runs\content\\runs\detect\\train\weights\\best.pt')

# Open camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #print(results[0])

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
cv2.destroyAllWindows() """