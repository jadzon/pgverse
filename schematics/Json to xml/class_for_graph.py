import json

class Node:
    def __init__(self, id, x=0, y=0, width=50, height=30,
                 shape="rectangle", element_type="generic", pins=None):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shape = shape
        self.element_type = element_type
        self.pins = pins if pins is not None else []

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "shape": self.shape,
            "element_type": self.element_type,
            "pins": self.pins
        }

    @staticmethod
    def from_dict(data):
        return Node(
            id=data["id"],
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 50),
            height=data.get("height", 30),
            shape=data.get("shape", "rectangle"),
            element_type=data.get("element_type", "generic"),
            pins=data.get("pins", [])
        )

    def __str__(self):
        pin_str = ", ".join(p['name'] for p in self.pins)
        return (f"Node({self.id}): {self.element_type}, shape={self.shape}, "
                f"pos=({self.x}, {self.y}), size=({self.width}x{self.height}), pins=[{pin_str}]")

class SchematicGraph:
    def __init__(self, directed=False):
        self.nodes = {}  # id -> Node
        self.edges = {}  # id -> list of (neighbor_id, weight)
        self.directed = directed

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self.edges.setdefault(node.id, [])

    def add_edge(self, from_id, to_id, weight=1):
        self.edges[from_id].append((to_id, weight))
        if not self.directed:
            self.edges[to_id].append((from_id, weight))

    def to_dict(self):
        return {
            "directed": self.directed,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": self.edges
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @staticmethod
    def from_dict(data):
        graph = SchematicGraph(directed=data.get("directed", False))
        for node_data in data.get("nodes", []):
            node = Node.from_dict(node_data)
            graph.add_node(node)
        for from_id, connections in data.get("edges", {}).items():
            for to_id, weight in connections:
                if graph.directed or from_id < to_id:
                    graph.add_edge(from_id, to_id, weight)
        return graph

    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        return SchematicGraph.from_dict(data)

    def __str__(self):
        result = "Nodes:\n"
        for node in self.nodes.values():
            result += f"  {node}\n"
        result += "Edges:\n"
        for node_id, neighbors in self.edges.items():
            connections = ", ".join(f"{nid} (w: {w})" for nid, w in neighbors)
            result += f"  {node_id} -> {connections}\n"
        return result
