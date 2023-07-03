print("Test of YOLOv8")

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
predictions = model.predict(source="D:\\SAJAT\\00_ML\\MY PROJECTS\\People recognization\\test_pics\\test.jpg", conf=0.25)

res_plotted = predictions[0].plot()
cv2.imshow("YOLOv8 Inference", res_plotted)
