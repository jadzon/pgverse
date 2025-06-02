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
            circuitikz_structure = self.create_structure_for_circuit(connections, results)
    
            return circuitikz_structure
                

        else:
            boxes = self._detect_diagram(processed_image_path)
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
            "resistor": "resistor",
            "capacitor": "C",
            "inductor": "L",
            "diode": "D",
            "battery": "battery",
            "voltage_source": "V",
            "current_source": "I",
            "transistor": "transistor",
            "op_amp": "op amp",
            "ground": "ground",
            "node": "",  # Will be handled differently
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
        scale = 0.1  # Scaling factor
        image_height = circuitikz_data.get("image_height", 1000)  # Default if not provided

        latex = [
            "\\documentclass{standalone}",
            "\\usepackage[siunitx, RPvoltages]{circuitikz}",
            "\\begin{document}",
            "\\begin{circuitikz}"
        ]

        # Draw components with orientation
        for comp in circuitikz_data["components"]:
            x1, y1 = comp["start_point1"] if "start_point1" in comp else comp["position"]
            x2, y2 = comp["start_point2"] if "start_point2" in comp else comp["position"]
            # Flip y-coordinate using image height
            y1_flipped = image_height - y1 if image_height else y1
            x1_scaled = x1 * scale
            y1_scaled = y1_flipped * scale
            y2_flipped = image_height - y2 if image_height else y2
            x2_scaled = x2 * scale
            y2_scaled = y2_flipped * scale


            # Handle rotation
            rotate = f"rotate={comp['rotation']}" if comp.get("rotation", 0) != 0 else ""

            if comp["type"] == "ground":
                latex.append(f"  \\draw ({x1_scaled:.2f},{y1_scaled:.2f}) to[{comp['type']}] ({x2_scaled:..2f},{y2_scaled}) {{}};")
            elif comp["type"] == "generic":
                latex.append(f"  \\node[draw, minimum width={comp['width']*scale:.2f}cm, "
                             f"minimum height={comp['height']*scale:.2f}cm, {rotate}] "
                             f"at ({x1_scaled:.2f},{y1_scaled:.2f}) ({x2_scaled},{y2_scaled}) {{{comp['label']}}};")
            else:
                latex.append(f"  \\draw ({x1_scaled:.2f},{y1_scaled:.2f}) node[{comp['type']}, {rotate}] "
                             f"({x2_scaled},{y2_scaled}) {{{comp['label']}}};")

        # Draw connections using actual paths
        for conn in circuitikz_data["connections"]:
            if conn["path"]:
                path_str = ""
                for i, point in enumerate(conn["path"]):
                    px, py = point
                    # Flip y-coordinate
                    py_flipped = image_height - py if image_height else py
                    if i == 0:
                        path_str = f"({px*scale:.2f},{py_flipped*scale:.2f})"
                    else:
                        path_str += f" -- ({px*scale:.2f},{py_flipped*scale:.2f})"
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
            net_detector.visualize_connections()
            return net_detector.connections if hasattr(net_detector, "connections") else None
        finally:
            # Przywrócenie oryginalnych funkcji
            cv2.imshow = original_imshow
            cv2.waitKey = original_waitKey


def main():
    # Inicjalizacja i uruchomienie analizy schematu
    analyzer = SchematicAnalyzer(
        model_path="block_detector/models/handwritten.pt",
        results_folder="main_results",
        preprocess_enabled=False,
    )
    analyzer.analyze(image_path="img/test2.png")


if __name__ == "__main__":
    main()