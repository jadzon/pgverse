import cv2 as cv

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
    if len(image.shape) != 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    closed_contours = []
    for i, c in enumerate(contours):
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
            cv.drawContours(image, contours, i, (0, 0, 255), 2)
    else:
        cv.drawContours(image, contours, i, (0, 255, 0), 2)
    cv.imshow("Closed contours", image)
    cv.waitKey(0)
    return closed_contours

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

for image in load_images_from_folder("example_schematics"):
    find_closed_contours(image)

