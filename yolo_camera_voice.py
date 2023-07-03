import cv2
from ultralytics import YOLO
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')

def speaking(text):
    engine.say(text)   #say something
    engine.runAndWait() #run the engine and wait for it to finish
    

# Load the YOLOv8 model
model = YOLO('D:\SAJAT\\00_ML\MY PROJECTS\People recognization\\runs\content\\runs\detect\\train\weights\\best.pt')

greetings = [0, 0]

# Open camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

frame_qty = 0

# Loop through the video frames
while cap.isOpened() :
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        for result in results:
            classes = result.boxes.cls
            probabilities = result.boxes.conf
        
        for i, class_id in enumerate(classes):
            if int(class_id) == 0:
                if float(probabilities[i]) > 0.9:
                    print("Hello Rustam")
            if int(class_id) == 1:
                if float(probabilities[i]) > 0.9:
                    print("Hello Adam")

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        #frame_qty += 1
        print('\n\n\n NEXT \n\n\n')
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()