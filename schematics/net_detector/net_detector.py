import cv2
import numpy as np
import os


class NetDetector():
    def __init__(self,image:cv2.typing.MatLike,blocks):
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
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Apply binary thresholding
        _, binary = cv2.threshold(blurred,130, 255, cv2.THRESH_BINARY_INV)
        # Apply Zhang-Suen thinning algorithm
        cv2.ximgproc.thinning(binary)
        #show image
        cv2.imshow("Skeleton", binary)
        cv2.waitKey(0)
        self.skeletonized_image = binary
        self.find_connections()
    def find_connections(self):
        """
        Finds connections between blocks based on the skeletonized image.
        Connections are found by finding pixels around blocks, and then following pixels until we reach another block.
        """
        self.connections = {}
        for i, block in enumerate(self.blocks):
            # Get the coordinates of the block
            x1, y1, x2, y2 = block.xyxy[0].tolist()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            print("Block coordinates: ", x1, y1, x2, y2)
            # Find white pixels around the block
            starting_points = []
            for x in range(x1,x2):
                if self.skeletonized_image[y1-5,x] == 255:
                    print("Found white pixel at: ", x, y1, "Block:", i)
                    starting_points.append((x,y1))

                if self.skeletonized_image[y2+5,x] == 255:
                    print("Found white pixel at: ", x, y2, "Block:", i)
                    starting_points.append((x,y2))
            for y in range(y1,y2):
                if self.skeletonized_image[y,x1-5] == 255:
                    print("Found white pixel at: ", x1, y, "Block:", i)
                    starting_points.append((x1,y))
                if self.skeletonized_image[y,x2+5] == 255:
                    print("Found white pixel at: ", x2, y, "Block:", i)
                    starting_points.append((x2,y))
            #merge starting_points if points are close enough to each other
            print("Starting points: ", starting_points)

            # Find connections by following pixels until we reach another block
            connections = []
            for j, other_block in enumerate(self.blocks):
                if i == j:
                    continue
                # Get the coordinates of the other block
                x1_other, y1_other, x2_other, y2_other = other_block.xyxy[0].tolist()
                # Check if the blocks are connected
                if (x1 < x2_other and x2 > x1_other) and (y1 < y2_other and y2 > y1_other):
                    connections.append(j)
            self.connections[i] = connections
        
        print("Connections: ", self.connections)
        #show image
        cv2.imshow("Connections", self.skeletonized_image)
        cv2.waitKey(0)
        

