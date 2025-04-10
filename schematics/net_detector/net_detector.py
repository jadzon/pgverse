import cv2
import numpy as np
import os

class NetDetector():
    def __init__(self,image,blocks):
        self.image = image.copy()
        self.blocks = blocks
        self.results = {}
    def cut_out_blocks(self):
        """
        Cuts out YOLO results from the image based on the coordinates provided in self.blocks.
        """
        self.cut_out_image = self.image.copy()
        for i, block in enumerate(self.blocks):
            # Get the coordinates of the block
            x1, y1, x2, y2 = block.xyxy[0].tolist()

            # Cut out the block from the image
            cv2.rectangle(self.cut_out_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
            
        cv2.imshow("Cut Out Block", self.cut_out_image)
        cv2.waitKey(0)
        self.skeletonize()
        
    def skeletonize(self):
        """
        Converts the image to grayscale, applies Gaussian blur, and then performs Zhang.
        """
        self.skeletonized_image = self.cut_out_image.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(self.skeletonized_image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply binary thresholding
        _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
        # Apply Zhang-Suen thinning algorithm
        cv2.ximgproc.thinning(binary)
        #show image
        cv2.imshow("Skeleton", binary)
        cv2.waitKey(0)
        
