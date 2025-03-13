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
        