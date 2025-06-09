from block_detector.block_detector import BlockDetector
from net_detector.net_detector import NetDetector
#from text_extraction.text_extraction import TextExtractor
from text_extraction.text_extraction_noPreprocess import TextExtractor
import cv2
import os
from preprocessing.png_proc import process_image
import json

class SchematicAnalyzer:
    def __init__(self, model_path, text_detection_enabled=True, preprocess_enabled=True, results_folder="main_results"):
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
        if len(boxes) != 0:
            
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

    def create_structure_for_circuit(self, connections, blocks, image_height=None):
        # Initialize connections for all blocks
        for block_id in blocks:
            blocks[block_id].setdefault("connections", [])

        # Add connections with proper node handling
        for block_id, block_connections in connections.items():
            if block_id not in blocks:
                # Preserve special node types if detected
                block_type = "node"
                if "block" in block_connections:
                    block_type = block_connections["block"]
                blocks[block_id] = {"block": block_type, "connections": []}

            for connected_block, start_point, path in block_connections:
                blocks[block_id]["connections"].append({
                    "block": connected_block,
                    "start_point": start_point,  # Start point of the connection
                    "path": path,
                    "path_length": len(path)
                })

        # Create circuitikz data structure
        circuitikz_data = {
            "components": [],
            "connections": [],
            "image_height": image_height  # Store for coordinate flip
        }

        # Enhanced component mapping
        component_mapping = {
            "ac_source": "sV",          # Sinusoidal voltage source
            "bjt": "npn",               # Default to NPN transistor
            "battery": "battery",
            "capacitor": "C",
            "current_source": "I",
            "dc_source": "V",           # DC voltage source
            "dep_current_source": "american controlled current source",
            "dep_dc_source": "american controlled voltage source",
            "diode": "D",
            "ground": "ground",
            "inductor": "L",
            "mosfet": "nmos",           # Default to NMOS
            "node": "",                 # Will be handled differently
            "opamp": "op amp",
            "resistor": "resistor",
            "resistor_box": "resistor",
            "voltage_source": "V",
            "zener_diode": "zDo",       # Zener diode
            "object": "generic",        # Generic component
            # Add more as needed
        }
        
        # Process components
        for block_id, block_data in blocks.items():
            comp_type = block_data["block"].lower()

            # Skip pure connection nodes (but keep grounds)
            if comp_type == "node":
                continue

            # Get circuitikz equivalent
            circuitikz_type = component_mapping.get(comp_type, "generic")

            # Extract position and dimensions from coordinates
            x1, y1 = block_data["coordinates"]["x1"], block_data["coordinates"]["y1"]
            x2, y2 = block_data["coordinates"]["x2"], block_data["coordinates"]["y2"]
            width = x2 - x1
            height = y2 - y1
            
            # Default center position (will be overridden if connection points are found)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Find connection starting points
            starting_points = []
            if "connections" in block_data and block_data["connections"]:
                for conn in block_data["connections"]:
                    if "start_point" in conn and conn["start_point"]:
                        starting_points.append(conn["start_point"])
            
            # If we have at least one starting point, use it to determine component position
            start_point1 = None
            start_point2 = None
            
            if len(starting_points) > 0:
                start_point1 = starting_points[0]
                
                # If we have multiple starting points, use the second one too
                if len(starting_points) > 1:
                    # Find the point furthest from the first one to get good component orientation
                    max_distance = 0
                    for point in starting_points[1:]:
                        dist = ((point[0] - start_point1[0])**2 + (point[1] - start_point1[1])**2)**0.5
                        if dist > max_distance:
                            max_distance = dist
                            start_point2 = point
                
                # If we only found one point, create a second point based on component dimensions
                if start_point2 is None:
                    # Create a second point in the opposite direction from the component center
                    dx = start_point1[0] - center_x
                    dy = start_point1[1] - center_y
                    if abs(dx) > abs(dy):  # Horizontal orientation
                        start_point2 = [center_x - dx, start_point1[1]]
                    else:  # Vertical orientation
                        start_point2 = [start_point1[0], center_y - dy]
                
                # Calculate new center point based on the two starting points
                center_x = (start_point1[0] + start_point2[0]) / 2
                center_y = (start_point1[1] + start_point2[1]) / 2

            # Determine orientation based on starting points if available
            if start_point1 and start_point2:
                dx = start_point2[0] - start_point1[0]
                dy = start_point2[1] - start_point1[1]
                
                if abs(dx) > abs(dy):  # Horizontal orientation
                    rotation = 0
                else:  # Vertical orientation
                    rotation = 90
            else:
                # Fallback to size-based orientation detection
                rotation = 0
                if width > height * 1.5:  # Horizontal component
                    rotation = 0
                elif height > width * 1.5:  # Vertical component
                    rotation = 90

            # Extract label
            label = ""
            if "texts" in block_data and block_data["texts"]:
                if isinstance(block_data["texts"], list):
                    # Take first text element if it exists
                    if block_data["texts"]:
                        text_item = block_data["texts"][0]
                        label = text_item[0] if isinstance(text_item, tuple) else str(text_item)
                else:
                    label = str(block_data["texts"])

            # Special handling for grounds
            if "ground" in comp_type:
                circuitikz_type = "ground"
                # Grounds are typically at connection points
                rotation = -90  # Standard ground orientation
            
            # Create the component with the starting points
            component_data = {
                "id": block_id,
                "type": circuitikz_type,
                "position": [center_x, center_y],
                "rotation": rotation,
                "width": width,
                "height": height,
                "label": label
            }
            
            # Add starting points if available
            if start_point1:
                component_data["start_point1"] = start_point1
            if start_point2:
                component_data["start_point2"] = start_point2
                
            circuitikz_data["components"].append(component_data)

        # Process connections
        connection_id = 0
        for block_id, block_data in blocks.items():
            for connection in block_data.get("connections", []):
                target_id = connection["block"]
                path = connection.get("path", [])

                # Only add each connection once
                if not any(conn["path"] == path for conn in circuitikz_data["connections"]):
                    circuitikz_data["connections"].append({
                        "id": connection_id,
                        "from": block_id,
                        "to": target_id,
                        "path": path
                    })
                    connection_id += 1

        # Export to files
        output_folder = os.path.join(self.results_folder, "circuitikz")
        os.makedirs(output_folder, exist_ok=True)

        json_path = os.path.join(output_folder, "circuit.json")
        with open(json_path, 'w') as f:
            json.dump(circuitikz_data, f, indent=2)

        latex_path = os.path.join(output_folder, "circuit.tex")
        with open(latex_path, 'w') as f:
            f.write(self._generate_circuitikz_code(circuitikz_data))

        print(f"CircuitTikZ files generated in {output_folder}")
        return blocks

    def _generate_circuitikz_code(self, circuitikz_data):
        """Generate LaTeX code with proper coordinate conversion"""
        image_height = circuitikz_data.get("image_height", 1000)  # Default if not provided
        
        # Calculate dynamic scale based on image size
        # Target width for the output diagram in cm
        target_width_cm = 15.0
        
        # Find the maximum x-coordinate in the circuit to determine the actual width
        max_x = 0
        for comp in circuitikz_data["components"]:
            if "start_point1" in comp:
                max_x = max(max_x, comp["start_point1"][0])
            if "start_point2" in comp:
                max_x = max(max_x, comp["start_point2"][0])
            max_x = max(max_x, comp["position"][0])
        
        for conn in circuitikz_data["connections"]:
            for point in conn.get("path", []):
                max_x = max(max_x, point[0])
        
        # Calculate scale factor (make sure we don't divide by zero)
        scale = target_width_cm / max_x if max_x > 0 else 0.01

        latex = [
            "\\documentclass{standalone}",
            "\\usepackage[siunitx, RPvoltages]{circuitikz}",
            "\\begin{document}",
            f"% Dynamic scale factor: {scale:.5f} (target width: {target_width_cm}cm)",
            "\\begin{circuitikz}"
        ]

        # Draw components with orientation
        for comp in circuitikz_data["components"]:
            x1, y1 = comp["start_point1"] if "start_point1" in comp else comp["position"]
            x2, y2 = comp["start_point2"] if "start_point2" in comp else comp["position"]
            # Flip y-coordinate using image height
            y1_flipped = image_height - y1 if image_height else y1
            x1_scaled = round(x1 * scale,1)
            y1_scaled = round(y1_flipped * scale,1)
            y2_flipped = image_height - y2 if image_height else y2
            x2_scaled = round(x2 * scale,1)
            y2_scaled = round(y2_flipped * scale,1)

            # Handle rotation
            rotate = f"rotate={comp['rotation']}" if comp.get("rotation", 0) != 0 else ""

            if comp["type"] == "ground":
                latex.append(f"  \\draw ({x1_scaled:.1f},{y1_scaled:.1f}) to[{comp['type']}] ({x2_scaled:.1f},{y2_scaled:.1f}) {{}};")
            elif comp["type"] == "generic":
                latex.append(f"  \\node[draw, minimum width={comp['width']*scale:.1f}cm, "
                             f"minimum height={comp['height']*scale:.1f}cm] "
                             f"at ({x1_scaled:.1f},{y1_scaled:.1f}) ({x2_scaled:.1f},{y2_scaled:.1f}) {{{comp['label']}}};")
            else:
                latex.append(f"  \\draw ({x1_scaled:.1f},{y1_scaled:.1f}) to[{comp['type']}, l = {{{comp['label']}}}] "
                             f"({x2_scaled:.1f},{y2_scaled:.1f});")

        # Draw connections using actual paths
        for conn in circuitikz_data["connections"]:
            if conn["path"]:
                path_str = ""
                for i, point in enumerate(conn["path"]):
                    px, py = point
                    # Flip y-coordinate
                    py_flipped = image_height - py if image_height else py
                    if i == 0:
                        path_str = f"({px*scale:.1f},{py_flipped*scale:.1f})"
                    else:
                        path_str += f" -- ({px*scale:.1f},{py_flipped*scale:.1f})"
                latex.append(f"  \\draw {path_str};")
            else:  # Fallback to straight line
                latex.append(f"  \\draw ({conn['from']}) -- ({conn['to']});")

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
            # Wykrywanie połączeń
            net_detector.find_connections()
            #net_detector.visualize_connections()
            return net_detector.connections if hasattr(net_detector, "connections") else None
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
            "\\begin{document}",
            f"% Dynamic scale factor: {scale:.5f} (target width: {target_width_cm}cm)",
            "\\begin{tikzpicture}[",
            "    block/.style={rectangle, draw, minimum width=2cm, minimum height=1cm, text centered},",
            "    arrow/.style={->, >=stealth, thick},",
            "    terminator/.style={draw, ellipse, minimum width=2cm, minimum height=1cm, text centered},",
            "    line/.style={draw},",
            "    decision/.style={diamond, draw, aspect=2, text centered},",
            "    data/.style={trapezium, trapezium left angle=70, trapezium right angle=110, draw, minimum width=2cm, minimum height=1cm, text centered},",
            "    text/.style={font=\\normalsize}",
            "]"
        ]

        # Map YOLO class IDs to TikZ styles
        style_mapping = {
            0: "arrow",
            1: "terminator",
            2: "arrow",
            3: "decision",
            4: "text",
            5: "data"
        }

        # Process all non-arrow elements first
        node_boxes = {}  # Store objects with their coordinates for arrow connection
        arrow_boxes = {} # Store arrows for later processing
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls)
            style = style_mapping.get(class_id, "block")
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
            
            # Handle different element types
            if class_id == 4:  # Text element
                # For detected text blocks, extract text directly from text_results
                text_content = ""
                for text in text_results:
                    txt_x1, txt_y1, txt_x2, txt_y2 = map(int, text["coords"])
                    # Check if text coordinates overlap with this box
                    if (abs(txt_x1 - x1) < 10 and abs(txt_y1 - y1) < 10 and
                        abs(txt_x2 - x2) < 10 and abs(txt_y2 - y2) < 10):
                        text_content = text["text"]
                        break
                
                # If no direct match found, use the text itself
                if not text_content:
                    for text in text_results:
                        if (x1 <= int(text["coords"][0]) <= x2 and 
                            y1 <= int(text["coords"][1]) <= y2):
                            text_content = text["text"]
                            break
                
                # Add text node directly to the diagram
                latex.append(f"  \\node[text] at ({center_x_scaled},{center_y_scaled}) {{{text_content}}};")
                
            elif style == "arrow":
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
            
            # Check if this text is already inside a block
            is_inside_block = False
            for node_id, node in node_boxes.items():
                if (node['x1'] <= txt_x1 <= node['x2'] and 
                    node['y1'] <= txt_y1 <= node['y2'] and
                    node['x1'] <= txt_x2 <= node['x2'] and
                    node['y1'] <= txt_y2 <= node['y2']):
                    # Add text to the block
                    latex.append(f"  \\node at ({node['center_x_scaled']},{node['center_y_scaled']}) {{{text['text']}}};")
                    is_inside_block = True
                    break
            
            # If text isn't inside any block, add it as a standalone text
            if not is_inside_block:
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
                latex.append(f"  \\draw[arrow] (box{source_block}) -- (box{dest_block});")

        # Close the TikZ picture and document
        latex.append("\\end{tikzpicture}")
        latex.append("\\end{document}")

        return "\n".join(latex)

def main():
    # Inicjalizacja i uruchomienie analizy schematu
    analyzer = SchematicAnalyzer(
        model_path="block_detector/models/handwritten.pt",
        results_folder="main_results",
        preprocess_enabled=False,
    )
    analyzer.analyze(image_path="img/test4.png")


if __name__ == "__main__":
    main()