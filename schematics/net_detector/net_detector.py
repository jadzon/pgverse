import cv2
import numpy as np
import os
from collections import Counter, deque
#TODO: Dynamic offsets

class NetDetector():
    def __init__(self,image:cv2.typing.MatLike,blocks,text_blocks,build_nodes=True):
        self.image = image.copy()
        self.blocks = blocks
        self.text_blocks = text_blocks
        self.results = {}
        self.build_nodes = build_nodes
        # Calculate image dimensions for scaling
        self.img_height, self.img_width = self.image.shape[:2]
        
    def _calculate_cut_offset(self, width, height, block_type=None):
        """
        Calculate dynamic offset for cutting out blocks based on block size.
        
        Args:
            width: Block width
            height: Block height
            block_type: Type/class of the block
            
        Returns:
            Appropriate offset value
        """
        # Base offset on minimum dimension (width or height)
        min_dimension = min(width, height)
        
        # Scale based on image size
        img_scale = min(1.0, max(0.3, (self.img_width + self.img_height) / 2000))
        
        # Set minimum and scale by block size
        min_offset = 3
        offset = max(min_offset, min_dimension * 0.05 * img_scale)
        
        # Adjust based on block type if needed
        if block_type is not None:
            # Add type-specific adjustments here
            if block_type == 0:  # For example, if block_type 0 needs larger offset
                offset *= 1.5
                
        return int(offset)
    
    def _calculate_edge_offset(self, width, height, block_type=None):
        """
        Calculate dynamic offset for edge detection based on block size.
        
        Args:
            width: Block width
            height: Block height
            block_type: Type/class of the block
            
        Returns:
            Appropriate edge offset value
        """
        # Edge offset should be larger than cut offset
        base_offset = self._calculate_cut_offset(width, height, block_type)
        
        # Scale for edge detection (slightly larger than cut offset)
        edge_scale = 2.5
        
        # Adjust based on block type
        if block_type is not None:
            # Nodes might need smaller offsets
            if block_type == 12:  # If it's a node
                edge_scale = 1.5
            # Small components need larger relative offsets
            elif min(width, height) < 20:
                edge_scale = 2.5
                
        return int(base_offset * edge_scale)

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
        print(f"Blocks starting points: {self.blocks_starting_points}")

        contours = cv2.findContours(self.skeletonized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        #draw contours on the cut out image
        contours = self.merge_intersecting_contours(contours)
        self.draw_random_color_contours(contours, text="Contours1")
       
        accepted_contours_ids = {-1: None}  # Set to store accepted contour IDs as keys and list of connected block IDs as values
        # Check if starting points were found in countours
        
        for i, block in enumerate(self.blocks):
            self.connections[i] = []  # Initialize connections for the block
            print(f"Block {i}: {int(block.cls)}")
            if int(block.cls) == 12:  # Skip nodes
                continue
            for starting_points in self.blocks_starting_points[i]:
                for x, y in starting_points:
                    for contour_id,contour in enumerate(contours):
                        #Check if the starting point is in the contour
                        result = cv2.pointPolygonTest(contour, (x,y), True)
                        print(f"Block:{i}, Contour ID: {contour_id}, Starting Point: ({x},{y}), Result: {result}")
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
        path_endpoints = self.get_path_endpoints(contours, accepted_contours_ids)
        
        # Visualize the endpoints and paths
        for contour_id, path_data in path_endpoints.items():
            # Draw connection points in red
            for x, y, block_id in path_data['connection_points']:
                cv2.circle(self.cut_out_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.cut_out_image, f"B{block_id}", (x+5, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw all paths
            for path in path_data['paths']:
                # Use a random color for each path
                #color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
                
                # Draw lines following the contour
                for i in range(len(path)-1):
                    pt1, pt2 = path[i], path[i+1]
                    cv2.line(self.cut_out_image, pt1, pt2, (0,255,0), 2)
                
                # Draw points along the path
                for pt in path:
                    cv2.circle(self.cut_out_image, pt, 2, (0, 255, 0), -1)
        cv2.imshow("Contours2", self.cut_out_image)


            # Prepare results dictionary
        results = {
            "block_connections": self.connections.copy(),
            "connection_points": self.blocks_starting_points.copy(),
            "contours": {}
        }
        
        # Structure contour data
        for contour_id, path_data in path_endpoints.items():
            results["contours"][contour_id] = {
                "connected_blocks": accepted_contours_ids[contour_id],
                "paths": path_data['paths']
            }
        print(f"Starting points: {self.blocks_starting_points}")

        return results
    def merge_intersecting_contours(self, merged_contours):
        """
        Checks if contours intersect and merges them if they do.

        Args:
            contours: List of contours to check and merge

        Returns:
            List of merged contours
        """
        if not merged_contours:
            return []

        # Create a copy of contours to modify

        changes_made = True

        # Continue until no more merges are possible
        while changes_made:
            changes_made = False
            i = 0

            while i < len(merged_contours):
                j = i + 1
                while j < len(merged_contours):
                    # Create masks for both contours
                    mask1 = np.zeros(self.skeletonized_image.shape, dtype=np.uint8)
                    mask2 = np.zeros(self.skeletonized_image.shape, dtype=np.uint8)

                    #cv2.drawContours(mask1, [merged_contours[i]], 0, 255, 1)
                    #cv2.drawContours(mask2, [merged_contours[j]], 0, 255, 1)

                    # Check if masks overlap
                    intersection = cv2.bitwise_and(mask1, mask2)
                    

                    if np.any(intersection):
                        # Contours intersect, merge them
                        merged_contour = np.vstack([merged_contours[i], merged_contours[j]])
                        
                        hull = cv2.convexHull(merged_contour)
                        epsilon = 0.1*cv2.arcLength(hull,True)
                        hull = cv2.approxPolyDP(hull, epsilon, True) 
                        # Update the first contour with the merged result
                        merged_contours[i] = hull

                        # Remove the second contour
                        merged_contours.pop(j)

                        changes_made = True
                    else:
                        j += 1
                i += 1

        return merged_contours
    def cut_out_blocks(self):
        self.no_block_img = self.image.copy()
        for block in self.blocks:
            if block.cls == 12:  # Skip nodes
                continue
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            width, height = x2 - x1, y2 - y1
            block_type = int(block.cls)
            
            # Calculate dynamic offset for this block
            cut_offset = self._calculate_cut_offset(width, height, block_type)
            
            print(f"Block {block_type} size: {width}x{height}, cut_offset: {cut_offset}")
            
            cv2.rectangle(
                self.no_block_img,
                (x1 - cut_offset, y1 - cut_offset),
                (x2 + cut_offset, y2 + cut_offset),
                (255, 255, 255),  # Fill interior with white
                -1
            )
        cv2.imshow("Cut Out Blocks", self.no_block_img)
        cv2.waitKey(0)
        
    def cut_out_text(self):
        self.cut_out_image = self.no_block_img.copy()
        print(self.text_blocks)
        for block in self.text_blocks:
            x1, y1, x2, y2 = map(int, block["coords"])
            width, height = x2 - x1, y2 - y1
            
            # Calculate dynamic offset for this text block
            cut_offset = self._calculate_cut_offset(width, height)
            
            print(f"Text size: {width}x{height}, cut_offset: {cut_offset}")
            
            cv2.rectangle(
                self.cut_out_image,
                (x1 - cut_offset, y1 - cut_offset),
                (x2 + cut_offset, y2 + cut_offset),
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
            if block.cls == 12:  # Skip nodes
                continue
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            width, height = x2 - x1, y2 - y1
            block_type = int(block.cls)
            
            # Calculate dynamic edge offset for this block
            edge_offset = self._calculate_edge_offset(width, height, block_type)
            
            print(f"Block {i} type: {block_type}, size: {width}x{height}, edge_offset: {edge_offset}")
            
            # Calculate block center
            block_center_x = (x1 + x2) // 2
            block_center_y = (y1 + y2) // 2
            
            # See if block is bipole, dipole, or tripole
            class_type = "bipole" if block.cls in [0,2,3,4,5,6,7,8,14,15,16,17] else "dipole" if block.cls in [4, 5, 6] else "tripole" if block.cls in [1,11] else "other"

            # Define edges with their centers - using dynamic edge_offset
            edges = [
                {"range": (x1 - edge_offset, x2 + edge_offset), "const": y1 - edge_offset, "type": "top", "center": (block_center_x, y1 - edge_offset)},
                {"range": (x1 - edge_offset, x2 + edge_offset), "const": y2 + edge_offset, "type": "bottom", "center": (block_center_x, y2 + edge_offset)},
                {"range": (y1 - edge_offset, y2 + edge_offset), "const": x1 - edge_offset, "type": "left", "center": (x1 - edge_offset, block_center_y)},
                {"range": (y1 - edge_offset, y2 + edge_offset), "const": x2 + edge_offset, "type": "right", "center": (x2 + edge_offset, block_center_y)}
            ]
            
            detected_edges = []
            detected_points = []
            starting_points[i] = []
            
            # Process each edge
            for edge in edges:
                edge_points = []
                
                # Scan along the edge
                for pos in range(edge["range"][0], edge["range"][1] + 1):
                    # Get coordinates based on edge type
                    if edge["type"] in ["top", "bottom"]:
                        x, y = pos, edge["const"]
                    else:
                        x, y = edge["const"], pos
                    
                    # Validate bounds
                    if (0 <= x < self.skeletonized_image.shape[1] and 
                        0 <= y < self.skeletonized_image.shape[0]):
                        # Check if pixel is part of the skeleton
                        if self.skeletonized_image[y, x] == 255:
                            edge_points.append((x, y))
                
                # If points found on this edge, keep only the closest to center
                if edge_points:
                    # Calculate distances to edge center
                    edge_center = edge["center"]
                    distances = [((p[0] - edge_center[0])**2 + (p[1] - edge_center[1])**2) for p in edge_points]
                    closest_idx = distances.index(min(distances))
                    closest_point = edge_points[closest_idx]
                    
                    detected_points.append(closest_point)
                    detected_edges.append(edge["type"])
                    
                    # Visualize the closest point
                    cv2.circle(self.cut_out_image, closest_point, 1, (0, 0, 255), -1)
                    cv2.putText(self.cut_out_image, f"({closest_point[0]},{closest_point[1]})", 
                               (closest_point[0] + 5, closest_point[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            #check detected edges, if we have a bipole we need to have all horizontal or vertical edges
            if class_type == "bipole":
                edge_counter = Counter(detected_edges)
                #if sum of left-right edges are more than sum of top-bottom, bipole horizontal else vertical
                horizontal = edge_counter["left"] + edge_counter["right"]
                vertical = edge_counter["top"] + edge_counter["bottom"]
                if horizontal > vertical:
                    for j,edge in enumerate(detected_edges):
                        if edge in ["top","bottom"]: detected_points[j] = None
                else:
                    for j,edge in enumerate(detected_edges):
                        if edge in ["left","right"]: detected_points[j] = None
            elif class_type == "tripole":
                # Tripole should have one horizontal and two vertical edges or vice versa, and the odd one(base) out should be first
                if len(detected_edges) == 3:
                    edge_counter = Counter(detected_edges)
                    if edge_counter["top"] == 1 or edge_counter["bottom"] == 1:
                        # If top or bottom is the odd one out, it should be first
                        if detected_edges[0] in ["top", "bottom"]:
                            # Ensure the first point is the odd one out
                            detected_points = [detected_points[0]] + [p for p in detected_points[1:] if p is not None]
                        else:
                            # Otherwise, swap the first point with the odd one out
                            for j, edge in enumerate(detected_edges):
                                if edge in ["top", "bottom"]:
                                    detected_points[0], detected_points[j] = detected_points[j], detected_points[0]
                                    break
                    elif edge_counter["left"] == 1 or edge_counter["right"] == 1:
                        # If left or right is the odd one out, it should be first
                        if detected_edges[0] in ["left", "right"]:
                            # Ensure the first point is the odd one out
                            detected_points = [detected_points[0]] + [p for p in detected_points[1:] if p is not None]
                        else:
                            # Otherwise, swap the first point with the odd one out
                            for j, edge in enumerate(detected_edges):
                                if edge in ["left", "right"]:
                                    detected_points[0], detected_points[j] = detected_points[j], detected_points[0]
                                    break
            starting_points[i].append([point for point in detected_points if point is not None])

        cv2.imshow("Starting Points", self.cut_out_image)
        return starting_points

    def draw_random_color_contours(self, contours,text="Contours"):
        """
        Draws contours on the cut out image with random colors.
        
        Args:
            contours: List of contours to draw
        """
        colors = []
        for i in range(len(contours)):
            # Generate random BGR color
            color = (
                np.random.randint(0, 255),  # Blue
                np.random.randint(0, 255),  # Green
                np.random.randint(0, 255)   # Red
            )
            colors.append(color)

        # Draw each contour with its unique color
        for i, contour in enumerate(contours):
            cv2.drawContours(self.cut_out_image, [contour], 0, colors[i], 2)

            # Optionally add contour ID
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(self.cut_out_image, str(i), (cX, cY), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(text, self.cut_out_image)
    def get_path_endpoints(self, contours, accepted_contours_ids):
        """
        Gets endpoints and paths of contours that connect blocks.
        
        Args:
            contours: List of contours
            accepted_contours_ids: Dictionary mapping contour IDs to connected block IDs
            
        Returns:
            Dictionary mapping contour IDs to their endpoints and path points
        """
        path_endpoints = {}
        
        for contour_id in accepted_contours_ids.keys():
            if contour_id == -1 or accepted_contours_ids[contour_id] is None:
                continue
                
            contour = contours[contour_id]
            # Get the points where the contour intersects with blocks
            connection_points = []
            block_ids = accepted_contours_ids[contour_id]
            
            for block_id in block_ids:
                for starting_points in self.blocks_starting_points[block_id]:
                    for x, y in starting_points:
                            connection_points.append((x, y, block_id))
        
            # Flatten contour array and convert to list of points
            contour_points = [(point[0][0], point[0][1]) for point in contour]
            
            # If we have connection points, we need to follow the contour
            if len(connection_points) >= 2:
                # For each connection point, find the closest point on the contour
                connection_indices = []
                for x, y, _ in connection_points:
                    point = np.array([x, y])
                    # Find closest contour point
                    distances = [np.linalg.norm(point - np.array(p)) for p in contour_points]
                    closest_idx = np.argmin(distances)
                    connection_indices.append(closest_idx)
                
                # Create a graph representing the contour (points can connect to neighbors)
                graph = {}
                for i, point in enumerate(contour_points):
                    # Connect to next and previous points (circular)
                    prev_idx = (i - 1) % len(contour_points)
                    next_idx = (i + 1) % len(contour_points)
                    graph[i] = [prev_idx, next_idx]
                
                # Find paths between connection points
                paths = []
                seen_paths = set()  # Track unique paths
                
                for i in range(len(connection_indices)):
                    for j in range(i+1, len(connection_indices)):
                        start_idx = connection_indices[i]
                        end_idx = connection_indices[j]
                        
                        # Find shortest path on the contour
                        path_indices = self.find_shortest_path(graph, start_idx, end_idx, len(contour_points))
                        if path_indices:
                            # Filter points that are too close to each other
                            path_points = []
                            for idx in path_indices:
                                curr_point = contour_points[idx]
                                if not path_points or curr_point not in path_points:
                                    path_points.append(curr_point)
                            
                            # Convert to tuple for hashing
                            path_tuple = tuple(path_points)
                            # Also check reversed path to catch paths in opposite direction
                            path_tuple_rev = tuple(reversed(path_points))
                            
                            if path_tuple not in seen_paths and path_tuple_rev not in seen_paths:
                                paths.append(path_points)
                                seen_paths.add(path_tuple)
                
                path_endpoints[contour_id] = {
                    'connection_points': connection_points,
                    'contour_points': contour_points,
                    'paths': paths
                }
    
        return path_endpoints

    def find_shortest_path(self, graph, start, end, max_length):
        """Find shortest path in a graph using BFS."""
        if start == end:
            return [start]
            
        visited = set([start])
        queue = deque([(start, [start])])
        
        while queue:
            (vertex, path) = queue.popleft()
            
            # Don't explore paths longer than the contour length
            if len(path) > max_length:
                continue
                
            for next_vertex in graph[vertex]:
                if next_vertex == end:
                    return path + [next_vertex]
                if next_vertex not in visited and next_vertex not in path:
                    # Check if the next vertex is not too close to the last point in the path
                    
                    visited.add(next_vertex)
                    queue.append((next_vertex, path + [next_vertex]))
        
        return None


