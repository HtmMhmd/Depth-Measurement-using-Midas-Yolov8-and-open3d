from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Union

torch.cuda.empty_cache()
class WoodenBox:
    def __init__(self,
                 image: Union[np.ndarray, str] = None,  # Input image or image path
                 weights_path: str = 'c:/Users/LENOVO/Desktop/Depth Measurement/YoloModelFiles/best.pt',  # Path to YOLO model weights
                 mask_shape: int = 640,  # Shape of the mask to be generated
                 conf: float = 0.95  # Confidence threshold for YOLO model
                 ):
        """
        Initialize a WoodenBox object for detecting wooden boxes using a YOLO model.

        Args:
            image (ndarray or str): Input image or image path.
            weights_path (str): Path to the YOLO model weights.
            mask_shape (int): Shape of the mask to be generated.
            conf (float): Confidence threshold for the YOLO model.
        """

        # Store parameters
        self.image = image
        self.weights_path = weights_path
        self.mask_shape = mask_shape
        self.masks = None  # Binary masks
        self.conf = conf  # Confidence threshold
        self.box = None  # Minimum bounding box
        self.sucess = False
    
    def run(self):
        """
        Run the WoodenBox object to detect wooden boxes.

        Returns:
            np.ndarray or None: The minimum bounding box of the wooden box, or None if the detection was unsuccessful.
        """
        # Get masks using YOLO segmentation
        self.GetMask()
        
        # If the detection was successful, get the minimum bounding box
        if (self.sucess):
            self.GetMinBBox()
        
        # Return the minimum bounding box
        return self.box

        
    def GetMask(self):
        """
        This function uses YOLO segmentation to generate a list of binary masks.

        Returns:
            None
        """
        # Initialize YOLO model with given weights path
        model = YOLO(self.weights_path)
        
        # Perform YOLO segmentation on the input image with the specified confidence threshold
        results = model(self.image, conf=self.conf, verbose=False)[0]
        
        # Check if masks exist in the results
        if (results.masks is not None):
            # If masks exist, set success to True and store masks
            self.sucess = True
            self.masks = results.masks.data
            self.masks = np.array(self.masks, dtype='uint8')
        else:
            # If masks do not exist, set success to 0
            self.sucess = False


    def GetCorners(self):
        """
        This function returns the detected corners using Harris corner detection on a binary mask.

        Returns:
            np.ndarray: A binary image with the detected corners.
        """
        # Convert the binary mask to a float32 numpy array
        grayfloatimg = np.float32(self.masks[0])
        
        # Apply Harris corner detection to the grayscale image
        rscores = cv2.cornerHarris(grayfloatimg, 8, 5, 0.07)
        
        # Threshold the corner response
        rscores = cv2.dilate(rscores,None)
        threshold = 0.01*rscores.max()
        
        # Return the detected corners
        return (rscores>threshold)
    
    def GetMinBBox(self):
        '''
        This function returns the minimum (rotated) box corners using a binary mask.

        Returns:
            np.ndarray: Array of shape (4, 2) representing the corners of the minimum box.
        '''
        # Find the contours of the mask using external retrieval mode and simplified approximation
        contours, hierarchy  = cv2.findContours(self.masks[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort the contours in descending order based on their area
        contours = tuple(sorted(contours, key=cv2.contourArea, reverse=True))
        
        # Find the minimum area rectangle (minimum bounding box) for the largest contour
        rect = cv2.minAreaRect(contours[0])
        
        # Get the corners of the minimum bounding box
        self.box = np.intp(cv2.boxPoints(rect))
        self.box = np.intp(cv2.boxPoints(rect))
