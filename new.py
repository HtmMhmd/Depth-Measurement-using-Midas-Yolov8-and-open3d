from ultralytics import YOLO
import cv2
import torch
torch.cuda.empty_cache()
import numpy as np

x = [107, 231, 455]
y =[90 ,30, 20]

coefficients = np.polyfit(x,y, 1)
poly_function = np.poly1d(coefficients)
license_plate_detector = YOLO('c:/Users/LENOVO/Desktop/Depth Measurement/YoloModelFiles/best.pt')
print("[Model Loaded]")
print("[Camera Initialized]")
cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != 27:
    _, frame = cap.read()

    frame = frame[0:480, 0:480]
    print(frame.shape)
    print(frame.shape)
    license_plates = license_plate_detector(frame, verbose=False, conf=0.9)[0]
    for license_plate in license_plates.boxes.data.tolist():
        
        x1, y1, x2, y2, score, class_id = license_plate
        x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
        if score > 0.8:
            dist = poly_function(x2-x1)
            cv2.putText(frame, f"{dist:.4f}", (50,50), 0, 1.2, (0,255,0), 2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow("frame", frame)
    # saver.save_frame(frame)

# saver.release()

cv2.destroyAllWindows()