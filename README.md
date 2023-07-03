# Office-people-recognization

## About the project
In this project I will realize a system which can detect employees at the workplace and automatically generate an excel table with time of arriving and leaving (last seen). With this system workers won't need to look into a camera to scan faces, it will use for input simply security cameras with good resolution. 

## Collecting data for training Neural Network
I created a [script](https://github.com/AGNworks/Office-people-recognization/blob/main/frame_gen.py) which helps in frame generation.With the help of a text to speech [library](https://pypi.org/project/pyttsx3/) for python we hear commands from the speaker when the frames starting to be saved and when the process is ended. 

## Data annotation
I made this step with the help of Roboflow. Created bounding boxes around the people and gave them the employee's name labels.

## Training the model
This step I made with the help of google colaboratory. Got the dataset straight from roboflow with the roboflow library installed in colab session. After succesfull training I downloaded the model and wrote two script to test the model, [one](https://github.com/AGNworks/Office-people-recognization/blob/main/yolo_test.py) to test on pictures and [another](https://github.com/AGNworks/Office-people-recognization/blob/main/yolo_video.py) to test with the local webcamera connected to the PC. 

## Integration of NN model 
This step is in process. I need to add functions that will get information from NN prediction and automaticly add data to the excel table which is saved on the local disk.
