from ultralytics import YOLO
import cv2
import torch
import numpy as np

torch.cuda.empty_cache()

class WoodenBox:
    def __init__(self,
               image        = None,
               weights_path = 'c:/Users/LENOVO/Desktop/Depth Measurement/YoloModelFiles/best.pt',
               mask_shape = 640,
               conf = 0.95):
        '''
        using yolo model trained weights this class detects woodenbox and return its bounding box

        @params
            image (ndarray 640,640) or (string): specify input image or image path
            conf                               : yolo model confidence
        '''
        
        self.image = image
        self.weights_path = weights_path
        self.mask_shape = mask_shape
        self.masks = None
        self.conf = conf
        self.box = None
        self.sucess = 0
    
    def run(self):
        self.GetMask()
        
        if (self.sucess):
            self.GetMinBBox()

        return self.box

        
    def GetMask(self):
        '''
        this function returns a list of binary masks using yolo segmentation

        '''
        model = YOLO(self.weights_path)
        results = model(self.image, conf = self.conf, verbose=False)[0]
        if (results.masks) != None:
            self.sucess = 1
            self.masks = results.masks.data
            self.masks =  np.array(self.masks, dtype='uint8')
        else:
            self.sucess = 0

    def GetCorners(self):
        '''
        this function returns the detected corners using harris corner detection on a binary mask
        '''

        grayfloatimg = np.float32(self.masks[0])
        rscores = cv2.cornerHarris(grayfloatimg, 8, 5, 0.07)

        rscores = cv2.dilate(rscores,None)
        threshold = 0.01*rscores.max()

        return (rscores>threshold)
    
    def GetMinBBox(self):
        '''
        this function returns the minimum (rotated) box corners using a binary mask
        '''
        contours, hierarchy  = cv2.findContours(self.masks[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = tuple(sorted(contours, key=cv2.contourArea , reverse=True))

        rect = cv2.minAreaRect(contours[0])
        self.box = np.intp(cv2.boxPoints(rect))