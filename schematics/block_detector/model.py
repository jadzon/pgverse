import cv2

import numpy as np
import os
from png_proc import process_image
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height
class Connection:
    def __init__(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node
class Graph:
    def __init__(self):
        self.nodes = []
        self.connections = []
    def add_node(self, node):
        node.id = len(self.nodes)
        self.nodes.append(node)
    def add_connection(self, start_node, end_node):
        connection = Connection(start_node, end_node)
        self.connections.append(connection)
    def get_all_nodes(self):
        return self.nodes
    def get_all_connections(self):
        return self.connections
    
class Node:
    def __init__(self,id): 
        self.id = None
        self.type = None
        self.value = None
        self.unit = ""
        self.label = ""
        self.position = Position(0, 0)
        self.size = Size(0, 0)
    def set_position(self, x, y):
        self.position = Position(x, y)
    def set_size(self, width, height):
        self.size = Size(width, height)
    def set_type(self, type):
        self.type = type
    def set_value(self, value):
        self.value = value
    def set_unit(self, unit):
        self.unit = unit
    def set_label(self, label):
        self.label = label

def find_closed_contours(image):
    #https://stackoverflow.com/questions/22240746/recognize-open-and-closed-shapes-opencv/59748729
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess: Thresholding and morphological operations
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove thin wires

    # Find contours of components
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

for image in load_images_from_folder("example_schematics"):
    find_closed_contours(image)


