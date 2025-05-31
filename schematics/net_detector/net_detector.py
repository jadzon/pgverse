import cv2
import numpy as np
import os

OFFSET = 5  # Offset for the bounding box expansion

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
        """
        self.connections = {}
        self.nodes = {}  # Dictionary to store nodes (intersections)
        self.node_counter = len(self.blocks)  # Start node IDs after block IDs

        self.cut_out_blocks()
        self.cut_out_text()
        self.skeletonize()
        self.blocks_starting_points = self.find_starting_points()
        
        cv2.imshow("Connections", self.cut_out_image)
        cv2.waitKey(0)
        
        # Create a visited matrix to track explored pixels
        self.visited = np.zeros_like(self.skeletonized_image)
        
        # Create a temporary dictionary to store all paths
        temp_connections = {}
        
        # Important: Create a copy of keys to avoid dictionary size change during iteration
        blocks_to_process = list(self.blocks_starting_points.keys())
        
        # First pass: process all initial blocks
        for key in blocks_to_process:
            print(f"Block {key} starting point: ", self.blocks_starting_points[key])
            temp_connections[key] = {}  # Initialize connections dict for this block
            
            for starting_point in self.blocks_starting_points[key]:
                # Get the coordinates of the starting point
                x, y = starting_point
                
                # Start DFS from this point
                path = []
                end_block = self.dfs_follow_connection(x, y, key, path)
                
                if end_block is not None:
                    # We found a connection to another block
                    if end_block not in temp_connections[key] or len(path) < len(temp_connections[key][end_block]):
                        temp_connections[key][end_block] = path
                        print(f"Found connection: Block {key} -> Block {end_block}, path length: {len(path)}")
        
        # Second pass: process any newly created nodes
        new_nodes = [k for k in self.blocks_starting_points.keys() if k not in blocks_to_process]
        while new_nodes:
            current_nodes = new_nodes
            new_nodes = []
            
            for node_id in current_nodes:
                if node_id not in temp_connections:
                    temp_connections[node_id] = {}
                    
                for starting_point in self.blocks_starting_points[node_id]:
                    x, y = starting_point
                    path = []
                    end_block = self.dfs_follow_connection(x, y, node_id, path)
                    
                    if end_block is not None and end_block != node_id:
                        if end_block not in temp_connections[node_id] or len(path) < len(temp_connections[node_id][end_block]):
                            temp_connections[node_id][end_block] = path
                            print(f"Found connection: Node {node_id} -> Block/Node {end_block}, path length: {len(path)}")
                        
                        # Check if this created new nodes that need processing
                        latest_nodes = [k for k in self.blocks_starting_points.keys() 
                                      if k not in blocks_to_process and k not in current_nodes and k not in new_nodes]
                        new_nodes.extend(latest_nodes)
        
        # Convert the temporary dictionary to the final connections format
        for source_id, connections in temp_connections.items():
            self.connections[source_id] = [(dest_id, path) for dest_id, path in connections.items()]
        
        # Print all connections
        print("\nFinal connections (only shortest paths):")
        for block_id, connections in self.connections.items():
            for connected_block, path in connections:
                print(f"  Block {block_id} -> Block {connected_block}, path length: {len(path)}")
        print(self.connections)
        cv2.imshow("Connections", self.cut_out_image)
        cv2.waitKey(0)
        return self.connections

    def dfs_follow_connection(self, start_x, start_y, source_id, path):
        stack = [(start_x, start_y, None)]  # (x, y, previous_direction)
        local_visited = set()  # Track visited pixels for this specific path

        while stack:
            x, y, prev_dir = stack.pop()

            # Skip if already visited in this DFS instance
            if (x, y) in local_visited:
                continue
            local_visited.add((x, y))

            # Check if we hit another block/node (excluding the source)
            found_block = self.check_if_block_hit(source_id, x, y)
            if found_block is not None and found_block != source_id:
                return found_block

            path.append((x, y))  # Add to global path

            # Visualization (optional)
            cv2.circle(self.cut_out_image, (x, y), 1, (0, 0, 255), -1)
            if len(path) % 10 == 0:
                cv2.imshow("DFS", self.cut_out_image)
                cv2.waitKey(1)

            # Get valid neighbors (prioritize straight directions)
            neighbors = []
            straight_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
            diag_dirs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

            # Prioritize continuing in the same direction if available
            if prev_dir:
                dx, dy = prev_dir
                straight_dirs = [prev_dir] + [d for d in straight_dirs if d != prev_dir]

            # Check all directions
            for dx, dy in straight_dirs + diag_dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.skeletonized_image.shape[1] and 
                    0 <= ny < self.skeletonized_image.shape[0]):
                    pixel_value = self.skeletonized_image[ny, nx]
                    # Only add if pixel is part of the skeleton and not visited locally
                    if pixel_value == 255 and (nx, ny) not in local_visited:
                        neighbors.append((nx, ny, (dx, dy)))

            # Handle intersections (node creation)
            if len(neighbors) > 2 and self.build_nodes:
                if not self.is_near_existing_node(x, y):
                    node_id = self.create_node(x, y)
                    return node_id

            # Push neighbors to stack (LIFO for depth-first)
            stack.extend(reversed(neighbors))

        return None  # Dead end

    def is_near_existing_node(self, x, y, distance=10):
        """Check if the current position is near an existing node"""
        for node_id, node_pos in self.nodes.items():
            node_x, node_y = node_pos
            if abs(x - node_x) < distance and abs(y - node_y) < distance:
                return True
        return False

    def create_node(self, x, y):
        """Create a new node at the intersection point"""
        node_id = self.node_counter
        self.node_counter += 1
        
        # Store node position
        self.nodes[node_id] = (x, y)
        
        # Create starting points for this node
        self.blocks_starting_points[node_id] = []
        
        # Define starting points around the node
        radius = 3
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if dx == 0 and dy == 0:
                    continue  # Skip the center point
                
                # Check bounds
                if 0 <= nx < self.skeletonized_image.shape[1] and 0 <= ny < self.skeletonized_image.shape[0]:
                    if self.skeletonized_image[ny, nx] == 255:
                        self.blocks_starting_points[node_id].append((nx, ny))
        
        # Draw the node for visualization
        cv2.circle(self.cut_out_image, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(self.cut_out_image, f"N{node_id}", (x+5, y+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Nodes", self.cut_out_image)
        cv2.waitKey(0)
        
        print(f"Created new node {node_id} at ({x}, {y})")
        return node_id

    def check_if_block_hit(self, skip_id, x, y, distance=15):
        for block_id, block in enumerate(self.blocks):
            if block_id == skip_id:
                continue
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            # Check if (x, y) is near the expanded block area
            if (x1 - distance <= x <= x2 + distance) and (y1 - distance <= y <= y2 + distance):
                return block_id
        return None

    def visualize_connections(self):
        """
        Visualize all found connections on the original image
        """
        vis_img = self.image.copy()
        
        # Draw all blocks
        for i, block in enumerate(self.blocks):
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"B{i}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw all nodes
        for node_id, (x, y) in self.nodes.items():
            cv2.circle(vis_img, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(vis_img, f"N{node_id}", (x+5, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw all connections
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for block_id, connections in self.connections.items():
            color_idx = block_id % len(colors)
            for connected_block, path in connections:
                # Draw the path
                for i in range(1, len(path)):
                    pt1 = path[i-1]
                    pt2 = path[i]
                    cv2.line(vis_img, pt1, pt2, colors[color_idx], 2)
        
        
        cv2.imshow("Circuit Connections", vis_img)
        cv2.waitKey(0)
        cv2.imwrite("circuit_connections.png", vis_img)

    def check_if_block_hit(self, skip_id, x, y, distance=10):
        for block_id, block in enumerate(self.blocks):
            if block_id == skip_id:
                continue
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            # Check if (x, y) is near the block's expanded bounding box
            expanded_x1 = x1 - distance
            expanded_y1 = y1 - distance
            expanded_x2 = x2 + distance
            expanded_y2 = y2 + distance
            if (expanded_x1 <= x <= expanded_x2 and 
                expanded_y1 <= y <= expanded_y2):
                return block_id
        return None

    def cut_out_blocks(self):
        self.no_block_img = self.image.copy()
        for block in self.blocks:
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            print(x1, y1, x2, y2)   
            cv2.rectangle(
                self.no_block_img,
                (x1 + OFFSET, y1 + OFFSET),
                (x2 - OFFSET, y2 - OFFSET),
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
                (x1 + OFFSET, y1 + OFFSET),
                (x2 - OFFSET, y2 - OFFSET),
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
            x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
            starting_points[i] = []
            
            # Check all 4 edges of the block
            edges = [
                (x1, x2, y1, "top"),    # Top edge (y = y1)
                (x1, x2, y2, "bottom"),  # Bottom edge (y = y2)
                (y1, y2, x1, "left"),    # Left edge (x = x1)
                (y1, y2, x2, "right")    # Right edge (x = x2)
            ]
            
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
                            starting_points[i].append((x, y))
                            # Visualize starting point
                            cv2.circle(self.cut_out_image, (x, y), 3, (0, 0, 255), -1)
            
            # Remove duplicates
            starting_points[i] = list(set(starting_points[i]))
        
        cv2.imshow("Starting Points", self.cut_out_image)
        cv2.waitKey(0)
        return starting_points






