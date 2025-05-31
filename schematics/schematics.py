from block_detector.block_detector import BlockDetector
from net_detector.net_detector import NetDetector
#from text_extraction.text_extraction import TextExtractor
from text_extraction.text_extraction_noPreprocess import TextExtractor
import cv2
import os
from preprocessing.png_proc import process_image


class SchematicAnalyzer:
    def __init__(self, model_path, text_detection_enabled=True, preprocess_enabled=True, results_folder="main_results"):
        self.text_detection_enabled = text_detection_enabled
        self.preprocess_enabled = preprocess_enabled
        self.block_detector = BlockDetector(model_path=model_path)

        self.model_path = model_path
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
        
        # Wykrywanie połączeń
        connections = self._detect_connections(image, boxes, text_results, nodes_exist)

        results = self.associate_text_with_blocks(text_results["blocks"], boxes)
        results = self.create_structure(connections, results)
        print(results)

        print(f"Wszystkie wyniki zostały zapisane w folderze {self.results_folder}")
        return {
            "processed_image": processed_image_path,
            "text_results": text_results,
            "boxes": boxes,
            "connections": connections
        }        
        
    def associate_text_with_blocks(self, text_results, block_results, distance_threshold=200):
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
            class_name = self.block_detector.model.names[int(block.cls)]
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

    def create_structure(self, connections, blocks):
        """
        Modyfikuje dictionary bloków aby dodać do nich połączenia.
        """
        # First, initialize connections list for each block
        for block_id in blocks:
            blocks[block_id]["connections"] = []
        
        # Then add connections
        for block_id, block_connections in connections.items():
            # Skip if block_id is not in blocks (e.g., it's a node)
            if block_id not in blocks:
                blocks[block_id] = {
                    "block": "node",
                    "connections": []
                }
                
            # Add each connection
            for connected_block, path in block_connections:
                blocks[block_id]["connections"].append({
                    "block": connected_block,
                    "path_length": len(path)
                })
        
        return blocks

    
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
    analyzer.analyze(image_path="img/test6.png")


if __name__ == "__main__":
    main()