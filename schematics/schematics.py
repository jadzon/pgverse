from block_detector.block_detector import BlockDetector
from net_detector.net_detector import NetDetector
#from text_extraction.text_extraction import TextExtractor
from text_extraction.text_extraction_noPreprocess import TextExtractor
import cv2
import numpy as np
import os
from collections import Counter, deque
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, tuple) and hasattr(o, '_fields'):  # for namedtuple
            return dict(zip(o._fields, o))
        return super(NumpyEncoder, self).default(o)

class SchematicAnalyzer:
    def __init__(self, text_detection_enabled=True, preprocess_enabled=True, results_folder="main_results"):
        self.text_detection_enabled = text_detection_enabled
        self.preprocess_enabled = preprocess_enabled
        self.block_detector = BlockDetector()
        self.results_folder = results_folder
        
        # Tworzenie struktury folderów na wyniki
        os.makedirs(self.results_folder, exist_ok=True)
        self.preprocessed_folder = os.path.join(self.results_folder, "preprocessed")
        self.text_results_folder = os.path.join(self.results_folder, "text_results")
        self.blocks_results_folder = os.path.join(self.results_folder, "blocks_results")
        self.connections_results_folder = os.path.join(self.results_folder, "connections_results")
        
        for folder in [self.preprocessed_folder, self.text_results_folder, 
                      self.blocks_results_folder, self.connections_results_folder]:
            os.makedirs(folder, exist_ok=True)

    def analyze(self, image_path):
        # Wczytaj obraz i wykonaj preprocessing
        image = cv2.imread(image_path)
        processed_image_path = self._preprocess_image(image_path, image)
        
        # Odczytaj przetworzony obraz
        image = cv2.imread(processed_image_path)
        
        # Wykrywanie tekstu
        text_results = self._detect_text(processed_image_path)
        
        # Wykrywanie bloków
        boxes, nodes_exist = self._detect_blocks(processed_image_path)
        #Modlimy się, że jeśli nie wykryto bloków, to wykryto diagram blokowy
        #W przeciwnym razie płaczemy
        if len(boxes) > 1:
            
            connections = self._detect_connections(image, boxes, text_results, nodes_exist)

            results = self.associate_text_with_circut_blocks(text_results["blocks"], boxes)
            circuitikz_code = self.create_structure_for_circuit(connections, results,image_height=image.shape[0])
    
            return circuitikz_code
                

        else:
            boxes = self._detect_diagram(processed_image_path)
            if len(boxes) == 0:
                print("Nie wykryto żadnych bloków ani diagramu blokowego.")
                return None
            tikz_code = self.generate_tikz_schematic(boxes, text_results["blocks"], image_height=image.shape[0])
            output_folder = os.path.join(self.results_folder, "tikz")
            os.makedirs(output_folder, exist_ok=True)
            latex_path = os.path.join(output_folder, "schematic.tex")
            with open(latex_path, 'w') as f:
                f.write(tikz_code)
            print(f"TikZ schematic file generated in {output_folder}")
        print(f"Wszystkie wyniki zostały zapisane w folderze {self.results_folder}")
           
        
    def associate_text_with_circut_blocks(self, text_results, block_results, distance_threshold=200):
        """
        Associate detected text with the detected blocks based on proximity.
        
        Args:
            text_results: List of detected text blocks
            block_results: List of detected component blocks
            distance_threshold: Maximum distance (in pixels) to associate text with a block
            
        Returns:
            List of blocks with associated text
        """
        # Initialize an empty list to store the associated results
        associated_results = {}

        # For each block, find the closest text within the threshold
        for i,block in enumerate(block_results):
            # Get block coordinates and calculate center point
            block_x1, block_y1, block_x2, block_y2 = map(int, block.xyxy[0].tolist())
            block_center_x = (block_x1 + block_x2) // 2
            block_center_y = (block_y1 + block_y2) // 2
            
            block_texts = []
            closest_texts = []
            
            # Check all text blocks
            for text in text_results:
                txt_x1,txt_y1,txt_x2,txt_y2 = map(int,text["coords"])
                
                # Calculate text center
                text_center_x = (txt_x1 + txt_x2) // 2
                text_center_y = (txt_y1 + txt_y2) // 2
                
                # Calculate Euclidean distance between block and text centers
                distance = ((block_center_x - text_center_x) ** 2 + 
                            (block_center_y - text_center_y) ** 2) ** 0.5
                
                # If text is inside the block, add it with distance 0
                if (block_x1 <= txt_x1 <= block_x2 and
                    block_y1 <= txt_y1 <= block_y2 and
                    block_x1 <= txt_x2 <= block_x2 and
                    block_y1 <= txt_y2 <= block_y2):
                    closest_texts.append((text, 0))
                # Otherwise check if it's within the distance threshold
                elif distance < distance_threshold:
                    closest_texts.append((text["text"], distance))
            
            # Sort texts by distance (closest first)
            closest_texts.sort(key=lambda x: x[1])
            if len(closest_texts) > 1:
                closest_text = closest_texts[:1]
            else:
                closest_text = ""
            class_name = self.block_detector.circuit_model.names[int(block.cls)]
            associated_results[i] = {
                'block': class_name,
                'coordinates': {
                    'x1': block_x1,
                    'y1': block_y1,
                    'x2': block_x2,
                    'y2': block_y2
                },
                'texts': closest_text  # Take the closest text only
            }
            
            # # Debug output
            # if block_texts:
            #     component_type = block.cls if hasattr(block, 'cls') else "Unknown"
            #     text_content = [text['text'] for text in block_texts]
            #     print(f"Component {component_type}: Associated with text: {text_content}")

        return associated_results
    
    def create_structure_for_circuit(self, connection_data, blocks, image_height=None):
        """
        Creates a structured representation of the circuit for CircuitTikZ.
        
        Args:
            connection_data: Dictionary with connection data from NetDetector.find_connections()
            blocks: Dictionary of detected blocks with their properties
            image_height: Height of the original image for coordinate conversion
            
        Returns:
            Processed blocks with CircuitTikZ structure
        """
        # Initialize connections for all blocks
        circuit = {}
        for block_id in blocks:
            print(f"Processing block {block_id} of type {blocks[block_id]['block']}")
            if blocks[block_id]["block"] == "Node":
                print(f"Skipping Node block {block_id} as it has no connections.")
                continue
            circuit[block_id] = {
                "block": blocks[block_id]["block"],
                "connections": connection_data["block_connections"][block_id],
                "coordinates": blocks[block_id]["coordinates"],
                "texts": blocks[block_id]["texts"],
                "connection_points": connection_data["connection_points"][block_id][0],
            }
                # Properly iterate through contours data and find connection points for this block

        # Save circuit to JSON
        output_folder = os.path.join(self.results_folder, "circuitikz")
        os.makedirs(output_folder, exist_ok=True)
        json_path = os.path.join(output_folder, "circuit_raw.json")
        with open(json_path, 'w') as f:
            json.dump(circuit, f, indent=2, cls=NumpyEncoder)
        print(f"Circuit data saved to {json_path}")
        # Store paths separately
        circuit["paths"] = []
        for contour_id, contour_data in connection_data["contours"].items():
            for path in contour_data["paths"]:
                circuit["paths"].append(path)       
        
        # Component mapping (existing code)
        component_mapping = {
            "ac_source": "sV",
            "bjt": "npn",
            "battery": "battery",
            "capacitor": "C",
            "current_source": "I",
            "dc_source": "V",
            "dep_current_source": "american controlled current source",
            "dep_dc_source": "american controlled voltage source",
            "diode": "D",
            "ground": "ground",
            "inductor": "L",
            "mosfet": "nmos",
            "node": "",
            "opamp": "op amp",
            "resistor": "resistor",
            "resistor_box": "resistor",
            "voltage_source": "V",
            "zener_diode": "zDo",
            "object": "generic",
        }
        
        circuitikz_data = {
            "components": [],
            "connections": [],
            "image_height": image_height if image_height else 1000,  # Default height if not provided
        }
        
        # Populate circuitikz_data with components
        for block_id, block_data in circuit.items():
            if block_id == "paths":
                continue  # Skip the paths entry
                
            component_type = block_data["block"]
            if component_type == "Node":
                continue
            circuitikz_type = component_mapping.get(str.lower(component_type), "generic")
            # Get component coordinates
            coords = block_data["coordinates"]
            x1, y1 = coords["x1"], coords["y1"]
            x2, y2 = coords["x2"], coords["y2"]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get component label from texts
            label = ""
            if block_data.get("texts") and len(block_data["texts"]) > 0:
                # Handle different text formats based on your structure
                if isinstance(block_data["texts"][0], tuple):
                    label = block_data["texts"][0][0]  # (text, distance) format
                elif isinstance(block_data["texts"][0], dict) and "text" in block_data["texts"][0]:
                    label = block_data["texts"][0]["text"]
                elif isinstance(block_data["texts"][0], str):
                    label = block_data["texts"][0]

            # Calculate orientation/rotation based on connection points
            component = {
                "id": block_id,
                "type": circuitikz_type,
                "position": (center_x, center_y),
                "label": label or "",
                "width": x2 - x1,
                "height": y2 - y1
            }
            if "connection_points" in block_data and len(block_data["connection_points"]) >= 2:
                points = block_data["connection_points"]
                print(f"Block {block_id} has connection points: {points}")
                if circuitikz_type in ["npn","nmos"]:
                    # For npn and nmos, first point is base
                    component["start_point1"] = points[0]  # Base
                    component["start_point2"] = points[1]  # Emitter/Source
                    component["start_point3"] = points[2]  # Collector/Drain
                else:
                    component["start_point1"] = points[0]
                    component["start_point2"] = points[1]
                    component["start_point3"] = points[2] if len(points) > 2 else None
            else:
                component["start_point1"] = None
                component["start_point2"] = None
                component["start_point3"] = None
            
            print(f"Component {block_id} connection points: {component.get('start_point1', 'N/A')} to {component.get('start_point2', 'N/A')}")
            circuitikz_data["components"].append(component)
        
        # Populate circuitikz_data with connections (paths)
        for i, path in enumerate(circuit["paths"]):
            connection = {
                "id": f"conn_{i}",
                "path": path
            }
            circuitikz_data["connections"].append(connection)
        
        # Export to files
        output_folder = os.path.join(self.results_folder, "circuitikz")
        os.makedirs(output_folder, exist_ok=True)

        latex_path = os.path.join(output_folder, "circuit.tex")
        with open(latex_path, 'w') as f:
            f.write(self._generate_circuitikz_code(circuitikz_data))

        print(f"CircuitTikZ files generated in {output_folder}")
        return blocks

    def _generate_circuitikz_code(self, circuitikz_data):
        """Generate LaTeX code with proper coordinate conversion"""
        image_height = circuitikz_data.get("image_height", 1000)  # Default if not provided
        
        # Calculate dynamic scale based on image size
        target_width_cm = 15.0
        
        # Find the maximum x-coordinate
        max_x = 0
        for comp in circuitikz_data["components"]:
            if comp["start_point1"] is not None:
                max_x = max(max_x, comp["start_point1"][0])
            if comp["start_point2"] is not None:
                max_x = max(max_x, comp["start_point2"][0])
            max_x = max(max_x, comp["position"][0])
        
        for conn in circuitikz_data["connections"]:
            for point in conn.get("path", []):
                max_x = max(max_x, point[0])
        
        # Calculate scale factor
        scale = target_width_cm / max_x if max_x > 0 else 0.01

        latex = [
            "\\documentclass{article}",
            "\\usepackage[siunitx, RPvoltages]{circuitikz}",
            "\\begin{document}",
            f"% Dynamic scale factor: {scale:.5f} (target width: {target_width_cm}cm)",
            "\\begin{circuitikz}"
        ]
        
        # Draw components with orientation
        for i,comp in enumerate(circuitikz_data["components"]):
            print()
            if comp["start_point1"] is None or comp["start_point2"] is None:
                x1, y1 = comp["position"]
                x2, y2 = comp["position"]
            else:
                x1, y1 = comp["start_point1"] 
                x2, y2 = comp["start_point2"]
            # Flip y-coordinate using image height
            y1_flipped = image_height - y1 if image_height else y1
            x1_scaled = round(x1 * scale,1)
            y1_scaled = round(y1_flipped * scale,1)
            y2_flipped = image_height - y2 if image_height else y2
            x2_scaled = round(x2 * scale,1)
            y2_scaled = round(y2_flipped * scale,1)
                

            #TODO: Handle tripoles (npn,pnp)
            rotate = ""
            print(f"Drawing component {comp['id']} of type {comp['type']} at ({x1_scaled:.1f},{y1_scaled:.1f}) to ({x2_scaled:.1f},{y2_scaled:.1f}) with rotation {rotate}")
            if comp["type"] == "ground":
                latex.append(f"  \\draw ({x1_scaled:.1f},{y1_scaled:.1f}) to[{comp['type']}] ({x2_scaled:.1f},{y2_scaled:.1f}) {{}};")
            elif comp["type"] in ["npn", "nmos"]:
                x3,y3 = comp["start_point3"]

                y3_flipped = image_height - y3 if image_height else y3
                x3_scaled = round(x3 * scale, 1)
                y3_scaled = round(y3_flipped * scale, 1)
                latex.append(f" \\draw ({x3_scaled:.1f},{y3_scaled:.1f}) node[{comp['type']}, anchor=B](Q{i}){{{comp["label"]}}}; ")
                latex.append(f"\\draw (Q{i}.B) -- ({x3_scaled:.1f},{y3_scaled:.1f});")
                latex.append(f"\\draw (Q{i}.E) -- ({x2_scaled:.1f},{y2_scaled:.1f});")
                latex.append(f"\\draw (Q{i}.C) -- ({x1_scaled:.1f},{y1_scaled:.1f});")
                
            elif comp["type"] == "generic":
                latex.append(f"  \\node[draw, minimum width={comp['width']*scale:.1f}cm, "
                             f"minimum height={comp['height']*scale:.1f}cm] "
                             f"at ({x1_scaled:.1f},{y1_scaled:.1f}) ({x2_scaled:.1f},{y2_scaled:.1f}) {{{comp['label']}}};")
            else:
                latex.append(f"  \\draw ({x1_scaled:.1f},{y1_scaled:.1f}) to[{comp['type']}, l = {{{comp['label']}}}, {rotate}] "
                             f"({x2_scaled:.1f},{y2_scaled:.1f});")
        # Draw connections using actual paths
        for conn in circuitikz_data["connections"]:
            if conn["path"]:
                # Sort path points to ensure they form a continuous line
                
                path_points = conn["path"]
                print(f"Processing connection {conn['id']} with {len(path_points)} path points")
                if len(path_points) >= 2:
                    path_str = ""
                    for i, point in enumerate(path_points):
                        px, py = point
                        # Flip y-coordinate
                        py_flipped = image_height - py if image_height else py
                        
                        # Format point
                        scaled_x = round(px * scale, 1)
                        scaled_y = round(py_flipped * scale, 1)
                        
                        if i == 0:
                            path_str = f"({scaled_x:.1f},{scaled_y:.1f})"
                        else:
                            path_str += f" -- ({scaled_x:.1f},{scaled_y:.1f})"
                    
                    # Add the path with appropriate styling
                    latex.append(f"  \\draw[color=black] {path_str};")
                    
            else:
                print(f"Connection {conn['id']} has no path points, using straight line.")
                # Fallback to straight line if no path points
                from_comp = None
                to_comp = None
                
                # Find component coordinates
                for comp in circuitikz_data["components"]:
                    if comp["id"] == conn["from"]:
                        from_comp = comp
                    if comp["id"] == conn["to"]:
                        to_comp = comp
                
                if from_comp and to_comp:
                    # Use component centers
                    from_x = from_comp["position"][0]
                    from_y = image_height - from_comp["position"][1] if image_height else from_comp["position"][1]
                    to_x = to_comp["position"][0]
                    to_y = image_height - to_comp["position"][1] if image_height else to_comp["position"][1]
                    
                    # Scale coordinates
                    from_x_scaled = round(from_x * scale, 1)
                    from_y_scaled = round(from_y * scale, 1)
                    to_x_scaled = round(to_x * scale, 1)
                    to_y_scaled = round(to_y * scale, 1)
                    
                    latex.append(f"  \\draw ({from_x_scaled:.1f},{from_y_scaled:.1f}) -- ({to_x_scaled:.1f},{to_y_scaled:.1f});")

        latex.append("\\end{circuitikz}")
        latex.append("\\end{document}")
        return "\n".join(latex)
    
    def _preprocess_image(self, image_path, image):
        """Preprocessuje obraz i zapisuje wyniki w odpowiednim folderze"""
        if not self.preprocess_enabled:
            return image_path
            
        processed = process_image(image)
        name, ext = os.path.splitext(os.path.basename(image_path))
        processed_image_path = os.path.join(self.preprocessed_folder, f"{name}_processed.png")
        
        # Zapisz różne wersje przetworzonego obrazu
        cv2.imwrite(processed_image_path, processed["enhanced_binary"])
        cv2.imwrite(os.path.join(self.preprocessed_folder, f"{name}_gray.png"), processed["gray"])
        cv2.imwrite(os.path.join(self.preprocessed_folder, f"{name}_denoised.png"), processed["denoised_gray"])
        cv2.imwrite(os.path.join(self.preprocessed_folder, f"{name}_edges.png"), processed["edges"])
        
        return processed_image_path
    
    def _detect_text(self, image_path):
        """Wykrywa tekst na obrazie i zapisuje wyniki"""
        if not self.text_detection_enabled:
            return {}
            
        text_extractor = TextExtractor(languages=['pl', 'en'], min_confidence=0.2)
        return text_extractor.extract_text(
            image_path=image_path,
            output_folder=self.text_results_folder,
            save_annotated=True,
            save_json=True
        )
    def _detect_diagram(self, image_path):
        """Wykrywa diagram blokowy na obrazie i zapisuje wyniki"""
        boxes = self.block_detector.detect_block_diagrams(
            image_path=image_path,
            conf_threshold=0.25
        )
        return boxes
    def _detect_blocks(self, image_path):
        """Wykrywa bloki na obrazie i zapisuje wyniki"""
       
        boxes, nodes_exist = self.block_detector.detect_electrical_symbols(
            image_path=image_path,
            conf_threshold=0.25
        )
        # Przeniesienie wykrytych bloków do odpowiedniego folderu
        name = os.path.splitext(os.path.basename(image_path))[0]
        detected_file = f"detected_{os.path.basename(image_path)}"
        
        if os.path.exists(detected_file):
            detected_img = cv2.imread(detected_file)
            output_path = os.path.join(self.blocks_results_folder, f"detected_{name}.png")
            cv2.imwrite(output_path, detected_img)
            try:
                os.remove(detected_file)
            except:
                pass
        
        return boxes, nodes_exist
    
    def _detect_connections(self, image, boxes, text_results, nodes_exist):
        """Wykrywa połączenia między blokami i zapisuje wyniki"""
        net_detector = NetDetector(
            image,
            boxes,
            text_results.get("blocks", []),
            build_nodes=not nodes_exist
        )
        
        # Zastąpienie funkcji wyświetlających okna funkcjami zapisującymi do plików
        name = os.path.basename(os.path.dirname(os.path.realpath(self.connections_results_folder)))
        name = os.path.join(os.path.basename(self.connections_results_folder), name)
        
        # Przechwycenie funkcji cv2 do wyświetlania okien
        original_imshow = cv2.imshow
        original_waitKey = cv2.waitKey
        
        def patched_imshow(window_name, img):
            output_path = os.path.join(
                self.connections_results_folder, 
                f"{os.path.basename(self.preprocessed_folder)}_{window_name}.png"
            )
            cv2.imwrite(output_path, img)
            
        def patched_waitKey(delay):
            return 1
            
        # Podmiana funkcji
        cv2.imshow = patched_imshow
        cv2.waitKey = patched_waitKey
        
        try:
            return net_detector.find_connections()
        finally:
            # Przywrócenie oryginalnych funkcji
            cv2.imshow = original_imshow
            cv2.waitKey = original_waitKey
    def generate_tikz_schematic(self, boxes, text_results, image_height=None):
        """
        Generate TikZ code for a general schematic diagram based on detected elements and text.
        """
        # Calculate dynamic scale based on image size
        target_width_cm = 15.0

        # Find the maximum x-coordinate to determine the actual width
        max_x = 0
        max_y = 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

        # Calculate scale factor
        scale = target_width_cm / max_x if max_x > 0 else 0.01

        # Start LaTeX document
        latex = [
            "\\documentclass{standalone}",
            "\\usepackage{tikz}",
            "\\usetikzlibrary{shapes.geometric, shapes.symbols, arrows, positioning, calc}",
            "\\usetikzlibrary {backgrounds}"
            "\\begin{document}",
            f"% Dynamic scale factor: {scale:.5f} (target width: {target_width_cm}cm)",
            "\\begin{tikzpicture}[",
            "    block/.style={rectangle, draw, minimum width=2cm, minimum height=1cm, text centered},",
            "    arrow/.style={->, >=stealth, thick},",
            "    terminator/.style={draw, ellipse, minimum width=2cm, minimum height=1cm, text centered},",
            "    line/.style={draw},",
            "    decision/.style={diamond, draw, aspect=2, text centered},",
            "    data/.style={trapezium, trapezium left angle=70, trapezium right angle=110, draw, minimum width=2cm, minimum height=1cm, text centered},",
            "    text/.style={font=\\normalsize},",
            "    process/.style={rectangle, draw, minimum width=2cm, minimum height=1cm,rounded corners=10mm, text centered},",
            "]"
        ]

        # Map YOLO class IDs to TikZ styles
        style_mapping = {
            0: "arrow",
            1: "terminator",
            2: "process",
            3: "decision",
            4: "text",
            5: "data",

        }

        # Process all non-arrow elements first
        node_boxes = {}  # Store objects with their coordinates for arrow connection
        arrow_boxes = {} # Store arrows for later processing
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls)
            style = style_mapping.get(class_id, "block")
            if style == "text": continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Flip y-coordinate using image height
            y1_flipped = image_height - y1 if image_height else y1
            y2_flipped = image_height - y2 if image_height else y2
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1_flipped + y2_flipped) / 2
            
            # Scale coordinates
            center_x_scaled = round(center_x * scale, 1)
            center_y_scaled = round(center_y * scale, 1)
            
            if style == "arrow":
                # Store arrow data for later processing
                width = x2 - x1
                height = y2 - y1
                
                arrow_boxes[i] = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'y1_flipped': y1_flipped, 'y2_flipped': y2_flipped,
                    'width': width, 'height': height,
                    'is_horizontal': width > height
                }
            else:
                # Process regular blocks
                width = x2 - x1
                height = y2 - y1
                width_scaled = round(width * scale, 1)
                height_scaled = round(height * scale, 1)
                
                # Add block without embedded text - we'll add text separately
                latex.append(f"  \\node[{style}, minimum width={width_scaled}cm, minimum height={height_scaled}cm] at ({center_x_scaled},{center_y_scaled}) (box{i}) {{}};")
                
                # Store node info for arrow connections
                node_boxes[i] = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'center_x': center_x, 'center_y': center_y,
                    'center_x_scaled': center_x_scaled, 'center_y_scaled': center_y_scaled,
                    'y1_flipped': y1_flipped, 'y2_flipped': y2_flipped
                }

        # Now add all text from text_results that isn't associated with blocks
        for text in text_results:
            txt_x1, txt_y1, txt_x2, txt_y2 = map(int, text["coords"])
            txt_center_x = (txt_x1 + txt_x2) / 2
            txt_center_y = image_height - (txt_y1 + txt_y2) / 2 if image_height else (txt_y1 + txt_y2) / 2
            
            # Scale coordinates
            txt_center_x_scaled = round(txt_center_x * scale, 1)
            txt_center_y_scaled = round(txt_center_y * scale, 1)
            
            latex.append(f"  \\node[text] at ({txt_center_x_scaled},{txt_center_y_scaled}) {{{text['text']}}};")

        # Build a connection graph and draw arrows (keep existing code)
        block_connections = {}
        
        # Check each arrow to find which blocks it connects
        for arrow_id, arrow in arrow_boxes.items():
            # Keep existing arrow processing code
            x1, y1 = arrow['x1'], arrow['y1']
            x2, y2 = arrow['x2'], arrow['y2']
            is_horizontal = arrow['is_horizontal']
            
            # Find blocks at the start and end of this arrow
            start_block = None
            end_block = None
            
            for node_id, node in node_boxes.items():

                # Check if arrow start point intersects with this block
                if (node['x1'] <= x1 <= node['x2'] and node['y1'] <= y1 <= node['y2']):
                    start_block = node_id
                    
                # Check if arrow end point intersects with this block
                if (node['x1'] <= x2 <= node['x2'] and node['y1'] <= y2 <= node['y2']):
                    end_block = node_id
            
            # If we couldn't find exact intersections, try to find the closest blocks
            if start_block is None or end_block is None:
                for node_id, node in node_boxes.items():
                    # For horizontal arrows
                    if is_horizontal:
                        # Start block (left side)
                        if start_block is None and x1 <= node['x2'] and abs(y1 - (node['y1'] + node['y2'])/2) < node['y2'] - node['y1']:
                            start_block = node_id
                        
                        # End block (right side)
                        if end_block is None and x2 >= node['x1'] and abs(y2 - (node['y1'] + node['y2'])/2) < node['y2'] - node['y1']:
                            end_block = node_id
                    
                    # For vertical arrows
                    else:
                        # Start block (top)
                        if start_block is None and y1 <= node['y2'] and abs(x1 - (node['x1'] + node['x2'])/2) < node['x2'] - node['x1']:
                            start_block = node_id
                        # End block (bottom)
                        if end_block is None and y2 >= node['y1'] and abs(x2 - (node['x1'] + node['x2'])/2) < node['x2'] - node['x1']:
                            end_block = node_id
            
            # Store the connection if we found both blocks
            if start_block is not None and end_block is not None and start_block != end_block:
                if start_block not in block_connections:
                    block_connections[start_block] = []
                
                # Add connection if it doesn't already exist
                if end_block not in block_connections[start_block]:
                    block_connections[start_block].append(end_block)
        
        # Draw arrows between connected blocks
        for source_block, destinations in block_connections.items():
            for dest_block in destinations:
                latex.append(f"\\begin{{scope}}[on background layer]"  
                             f"\\draw[arrow] (box{source_block}) -- (box{dest_block});"
                             f"\\end{{scope}}")
                            
        # Close the TikZ picture and document
        latex.append("\\end{tikzpicture}")
        latex.append("\\end{document}")

        return "\n".join(latex)

def main():
    # Inicjalizacja i uruchomienie analizy schematu
    analyzer = SchematicAnalyzer(
        results_folder="main_results",
        preprocess_enabled=False,
    )
    analyzer.analyze(image_path="img/test9.png")


if __name__ == "__main__":
    main()