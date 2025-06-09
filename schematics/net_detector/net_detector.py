import cv2
import numpy as np
import os
from collections import Counter
CUT_OFFSET = 5  # Offset for cutting out blocks
OFFSET = 10  # Offset for the bounding box expansion

class NetDetector():
    def __init__(self,image:cv2.typing.MatLike,blocks,text_blocks,build_nodes=True):
        self.image = image.copy()
        self.blocks = blocks
        self.text_blocks = text_blocks
        self.results = {}
        self.build_nodes = build_nodes
    def find_connections(self):
        """
        Main function of the class. It finds connections between blocks based on the skeletonized image.
        Connections are found by finding pixels around blocks, and then following pixels until we reach another block.
        connections are stored in the self.connections dictionary, where keys are block IDs and values are lists of connected block IDs.
        """
        self.connections = {}
        self.nodes = {}
        self.node_counter = len(self.blocks)  # Start node IDs after block IDs
        self.cut_out_blocks()
        self.cut_out_text()
        self.skeletonize()
        self.blocks_starting_points = self.find_starting_points()
        cv2.imshow("Connections", self.cut_out_image)
        cv2.waitKey(0)
        contours = cv2.findContours(self.skeletonized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        #draw contours on the cut out image
        cv2.drawContours(self.cut_out_image, contours, -1, (0, 0, 255), 1)
        cv2.imshow("Contours1", self.cut_out_image)
        accepted_contours_ids = {-1: None}  # Set to store accepted contour IDs as keys and list of connected block IDs as values
        # Check if starting points were found in countours
        for i, block in enumerate(self.blocks):
            self.connections[i] = []  # Initialize connections for the block
            if block.cls == 12:  # Skip nodes
                continue
            for starting_points in self.blocks_starting_points[i]:
                for x, y in starting_points:
                    for contour_id,contour in enumerate(contours):
                        if len(contour) < 3:
                            continue
                        #Check if the starting point is in the contour
                        result = cv2.pointPolygonTest(contour, (x,y), True)
                        print(f"result: {result}, point: ({x},{y})")
                        if result >= -20:
                            if contour_id in accepted_contours_ids.keys():
                                self.connections[i] += accepted_contours_ids[contour_id]
                                for accepted_id in accepted_contours_ids[contour_id]:
                                    self.connections[accepted_id]+= [i]
                            else:
                                accepted_contours_ids[contour_id] = []
                            accepted_contours_ids[contour_id] += [i]

        #Remove duplicate connections
        for i in self.connections.keys():
            self.connections[i] = list(set(self.connections[i]))
        selected_contours = [contours[i] for i in accepted_contours_ids]
        cv2.drawContours(self.cut_out_image, selected_contours, -1, (0, 255, 0), 1)
        cv2.imshow("Contours2", self.cut_out_image)
        print(f"Connections: {self.connections}")
        return {}
    def cut_out_blocks(self):
        self.no_block_img = self.image.copy()
        for block in self.blocks:
            if block.cls == 12 :  # Skip nodes
                continue
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            print(x1, y1, x2, y2)   
            cv2.rectangle(
                self.no_block_img,
                (x1 - CUT_OFFSET, y1 - CUT_OFFSET),
                (x2 + CUT_OFFSET, y2 + CUT_OFFSET),
                (255, 255, 255),  # Fill interior with white
                -1
            )
        cv2.imshow("Cut Out Blocks", self.no_block_img)
        cv2.waitKey(0)
    def cut_out_text(self):
        self.cut_out_image = self.no_block_img.copy()
        print(self.text_blocks)
        for block in self.text_blocks:
            x1, y1, x2, y2 = map(int,block["coords"])
            print(x1, y1, x2, y2)
            cv2.rectangle(
                self.cut_out_image,
                (x1 - CUT_OFFSET, y1 - CUT_OFFSET),
                (x2 + CUT_OFFSET, y2 + CUT_OFFSET),
                (255, 255, 255),  # Fill interior with white
                -1
            )
        cv2.imshow("Cut Out Text", self.cut_out_image)
        cv2.waitKey(0)
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
        skeleton = cv2.ximgproc.thinning(binary)
        #show image
        cv2.imshow("Skeleton", skeleton)
        cv2.waitKey(0)
        self.skeletonized_image = skeleton
    def find_starting_points(self):
        starting_points = {}
        for i, block in enumerate(self.blocks):
            if block.cls == 12 :  # Skip nodes
                continue
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            #See if block is bipole, dipole, or tripole
            class_type = "bipole" if block.cls in [1,2,3,4,5,6,7,8,14,15,16,17] else "dipole" if block.cls in [4, 5, 6] else "tripole" if block.cls in [7, 8] else "other"

            # Check all 4 edges of the block
            edges = [
                (x1 - OFFSET, x2 + OFFSET, y1 - OFFSET, "top"),    # Top edge (y = y1)
                (x1 - OFFSET, x2 + OFFSET, y2 + OFFSET, "bottom"),  # Bottom edge (y = y2)
                (y1 - OFFSET, y2 + OFFSET, x1 - OFFSET, "left"),    # Left edge (x = x1)
                (y1 - OFFSET, y2 + OFFSET, x2 + OFFSET, "right")    # Right edge (x = x2)
            ]
            detected_edges = []
            detected_points = []
            starting_points[i] = []
            for start, end, const, edge_type in edges:
                for pos in range(start, end + 1):
                    # Get coordinates based on edge type
                    if edge_type in ["top", "bottom"]:
                        x, y = pos, const
                    else:
                        x, y = const, pos
                    
                    # Validate bounds
                    if (0 <= x < self.skeletonized_image.shape[1] and 
                        0 <= y < self.skeletonized_image.shape[0]):
                        # Check if pixel is part of the skeleton
                        if self.skeletonized_image[y, x] == 255:
                            detected_points.append((x, y))
                            detected_edges.append(edge_type)
                            # Visualize starting point
                            cv2.circle(self.cut_out_image, (x, y), 1, (0, 0, 255), -1)
                            cv2.putText(self.cut_out_image, f"({x},{y})", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            #check detected edges, if we have a bipole we need to have all horizontal or vertical edges
            if class_type != "bipole":
                continue
            edge_counter = Counter(detected_edges)
            #if sum of left-right edges are more than sum of top-bottom, bipole horizontal else vertical
            horizontal = edge_counter["left"] + edge_counter["right"]
            vertical = edge_counter["top"] + edge_counter["bottom"]
            if horizontal > vertical:
                for j,edge in enumerate(detected_edges):
                    if edge in ["top,bottom"]: detected_points[j] = None
            else:
                for j,edge in enumerate(detected_edges):
                    if edge in ["left,right"]: detected_points[j] = None
            
            starting_points[i].append([point for point in detected_points if point != None])

        cv2.imshow("Starting Points", self.cut_out_image)
        cv2.waitKey(0)
        return starting_points
    def remove_intermiediate_point_in_path(self, path):
        """
        Remove intermediate points in the path that are not necessary.
        """
        if len(path) < 3:
            return path
        
        simplified_path = [path[0]]
        change_along_x = path[1][0] != path[0][0]

        for i in range(1, len(path) - 1):
            current = path[i]
            prev_point = path[i + 1]
            
            # Check if the current point is an intermediate point
            if change_along_x:
                if current[0] == simplified_path[-1][0]:    
                    # If the x-coordinate is the same as the last point, skip this point
                    continue
                else:
                    change_along_x = False
                    simplified_path.append(prev_point)
            else:
                if current[1] == simplified_path[-1][1]:    
                    # If the y-coordinate is the same as the last point, skip this point
                    continue
                else:
                    change_along_x = True
                    simplified_path.append(prev_point)
            simplified_path.append(current)
        simplified_path.append(path[-1])
        return simplified_path





