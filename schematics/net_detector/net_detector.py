import cv2
import numpy as np
import os



class NetDetector():
    def __init__(self,image:cv2.typing.MatLike,blocks,build_nodes=True):
        self.image = image.copy()
        self.blocks = blocks
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
        self.skeletonize()
        self.blocks_starting_points = self.find_starting_points()
        
        cv2.imshow("Connections", self.cut_out_image)
        cv2.waitKey(0)
        
        # Create a visited matrix to track explored pixels
        self.visited = np.zeros_like(self.skeletonized_image)
        
        # Iterate over the starting points and find connections
        for key, block_starting_point in self.blocks_starting_points.items():
            print(f"Block {key} starting point: ", block_starting_point)
            self.connections[key] = []  # Initialize connections list for this block
            
            for starting_point in block_starting_point:
                # Get the coordinates of the starting point
                x, y = starting_point
                
                # Start DFS from this point
                path = []
                end_block = self.dfs_follow_connection(x, y, key, path)
                
                if end_block is not None:
                    # We found a connection to another block
                    self.connections[key].append((end_block, path))
                    print(f"Found connection: Block {key} -> Block {end_block}")
        
        # Print all connections
        print("All connections found:")
        for block_id, connections in self.connections.items():
            for connected_block, path in connections:
                print(f"  Block {block_id} -> Block {connected_block}, path length: {len(path)}")
        
        return self.connections

    def dfs_follow_connection(self, x, y, source_block_id, path):
        """
        DFS algorithm to follow connections from a starting point to another block.
        
        Args:
            x, y: Starting pixel coordinates
            source_block_id: ID of the block where this search started
            path: List to store the path taken
            
        Returns:
            The ID of the block that was reached, or None if no block was reached
        """
        # Check if we've hit any block except the source
        target_block = self.check_if_block_hit(source_block_id, x, y)
        if target_block is not None:
            return target_block
        
        # Mark current pixel as visited
        self.visited[y, x] = 1
        path.append((x, y))
        
        # Draw the path for visualization
        vis_img = self.cut_out_image.copy()
        cv2.circle(vis_img, (x, y), 1, (0, 0, 255), -1)
        
        # Get neighboring pixels (8 directions)
        directions = [
            (0, -1),  # Up
            (1, -1),  # Up-Right
            (1, 0),   # Right
            (1, 1),   # Down-Right
            (0, 1),   # Down
            (-1, 1),  # Down-Left
            (-1, 0),  # Left
            (-1, -1)  # Up-Left
        ]
        
        # Count white neighboring pixels to check for intersection
        white_neighbors = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check bounds
            if 0 <= nx < self.skeletonized_image.shape[1] and 0 <= ny < self.skeletonized_image.shape[0]:
                if self.skeletonized_image[ny, nx] == 255:
                    white_neighbors += 1
        
        # If this is an intersection (more than 2 white neighbors) and build_nodes is True
        if white_neighbors > 2 and self.build_nodes and not self.is_near_existing_node(x, y):
            # Create a new node
            node_id = self.create_node(x, y)
            return node_id
        
        # Explore each direction
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < self.skeletonized_image.shape[1] and 0 <= ny < self.skeletonized_image.shape[0]:
                # If the pixel is white and not visited
                if self.skeletonized_image[ny, nx] == 255 and self.visited[ny, nx] == 0:
                    # Recursively follow this path
                    result = self.dfs_follow_connection(nx, ny, source_block_id, path)
                    if result is not None:
                        return result
        
        # If we've reached a dead end and didn't find a block
        return None

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
        cv2.waitKey(1)
        
        print(f"Created new node {node_id} at ({x}, {y})")
        return node_id

    def check_if_block_hit(self, skip_id, x, y, distance=5):
        """
        Checks if the pixel is near any block's starting point except the one with id skip_id.
        """
        for block_id, starting_points in self.blocks_starting_points.items():
            if block_id == skip_id:
                continue
                
            for sx, sy in starting_points:
                if abs(x - sx) < distance and abs(y - sy) < distance:
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

    def check_if_block_hit(self,skip_id,x, y):
        """
        Checks if the pixel is inside any starting point except the one with id skip_id.
        """
        for key, block_starting_point in self.blocks_starting_points.items():
            if key == skip_id:
                continue
            for starting_point in block_starting_point:
                # Get the coordinates of the starting point
                x1, y1 = starting_point
                # Check if the pixel is inside the block
                if x1 - 2 < x < x1 + 2 and y1 - 2 < y < y1 + 2:
                    return key
        return None
    def follow_pixel(self, x, y):
        """
        Follows the pixel until we reach another block or a black pixel (0).
        """
        connection = []
        while self.skeletonized_image[y, x] == 255:
            # Check the surrounding pixels (up, down, left, right)
            if self.skeletonized_image[y - 1, x] == 255:
                y -= 1
            elif self.skeletonized_image[y + 1, x] == 255:
                y += 1
            elif self.skeletonized_image[y, x - 1] == 255:
                x -= 1
            elif self.skeletonized_image[y, x + 1] == 255:
                x += 1
            else:
                break
        return connection
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
        _, binary = cv2.threshold(blurred,120, 255, cv2.THRESH_BINARY_INV)
        # Apply Zhang-Suen thinning algorithm
        cv2.ximgproc.thinning(binary)
        #show image
        cv2.imshow("Skeleton", binary)
        cv2.waitKey(0)
        self.skeletonized_image = binary
    def find_starting_points(self):
        """
        Finds starting points for each block for connections based on the skeletonized image.
        Starting points are found by finding white pixels around blocks
        """
        starting_points = {}
        for i, block in enumerate(self.blocks):
            # Get the coordinates of the block
            x1, y1, x2, y2 = block.xyxy[0].tolist()
            starting_points[i] = []
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            # Find white pixels around the block
            for x in range(x1,x2):
                if self.skeletonized_image[y1-5,x] == 255:
                    print("Found white pixel at: ", x, y1, "Block:", i)
                    starting_points[i].append((x,y1))
                    cv2.circle(self.cut_out_image, (x,y1-5), 5, (255, 0, 0), -1)
                if self.skeletonized_image[y2+5,x] == 255:
                    print("Found white pixel at: ", x, y2, "Block:", i)
                    starting_points[i].append((x,y2))
                    cv2.circle(self.cut_out_image, (x,y2+5), 5, (255, 0, 0), -1)
            for y in range(y1,y2):
                if self.skeletonized_image[y,x1-5] == 255:
                    print("Found white pixel at: ", x1, y, "Block:", i)
                    starting_points[i].append((x1,y))
                    cv2.circle(self.cut_out_image, (x1-5,y), 5, (255, 0, 0), -1)
                if self.skeletonized_image[y,x2+5] == 255:
                    print("Found white pixel at: ", x2, y, "Block:", i)
                    starting_points[i].append((x2,y))
                    cv2.circle(self.cut_out_image, (x2+5,y), 5, (255, 0, 0), -1)
            # Remove duplicates in range of 5 pixels
            starting_points[i] = list(set(starting_points[i]))
            # Merge starting points if they are close enough to each other
            starting_points[i]= [(x,y) for x,y in starting_points[i] if x1-5 < x < x2+5 and y1-5 < y < y2+5]
        return starting_points






