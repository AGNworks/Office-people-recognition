# It's a little script to collect video frames for further processing and preparing for NN dataset.
import cv2  
import pyttsx3
import matplotlib.pyplot as plt
import numpy as np
import time 

#Quantity of saved frames at once
frame_qty = 5

#initialize the engine
engine = pyttsx3.init()

#get the list of available languages
for voice in engine.getProperty('voices'):
    print(voice)

#set the voice to english
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')

def speaking(text):
    engine.say(text)   #say something
    engine.runAndWait() #run the engine and wait for it to finish


speaking("Hello! Let's create some frames!")

#reading from file the current number of the frame
with open ("imgnumb.txt", "r") as numbfile:
    imgnumb = int(numbfile.read())
print(imgnumb)

#open the camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

#set the resolution of the camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

#set the fps
#cap.set(cv2.CAP_PROP_FPS, 2)

#check if it was successfully opened
if not (cap.isOpened()):
    print("Could not open video device")

time.sleep(2)
speaking("Start")
#we can change how many frames we are saving for once
limit = imgnumb + frame_qty
#looping for needed quantity
while(imgnumb < limit): 
    success, frame = cap.read()
    if success:
        #save the frame
        cv2.imwrite(f'D:\SAJAT\\00_ML\MY PROJECTS\People recognization\\tes_pics\{imgnumb}.jpg', frame.astype('uint8'))
        print(f"{imgnumb}.jpg saved")
        imgnumb += 1

        #writing to the file the current number of the frame
        with open ("imgnumb.txt", "w") as numbfile: 
            numbfile.write(str(imgnumb))

        time.sleep(0.1)

speaking("We are done, thank you for your time")

cap.release()
cv2.destroyAllWindows()