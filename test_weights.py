from Wooden import *
import numpy as np

x = [107, 231, 455]
y =[90 ,30, 20]
coefficients = np.polyfit(x,y, 3)

poly_function = np.poly1d(coefficients)
wooden        = WoodenBox(conf= 0.8)

print("[Model Loaded]")
print("[Camera Initialized]")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cv2.waitKey(1) != 27:

    _, frame = cap.read()
    if _ :
        frame = frame[0:480, 0:480]
        print(frame.shape)

    wooden.image = frame
    bbox = wooden.run()

    if (wooden.sucess) :
        dist = poly_function((bbox[1] - bbox[0])[0])
        cv2.putText(frame, f"{dist:.4f}", (50,50), 0, 1.2, (0,255,0), 2)
        frame = cv2.drawContours(frame, [bbox], -1, (0, 255, 0), 1)
    else :
        print('NOTHING IS DETECTED')

    cv2.imshow("frame", frame)

cv2.destroyAllWindows()