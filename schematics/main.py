from block_detector.block_detector import BlockDetector
from net_detector.net_detector import NetDetector
#from text_extraction.text_extraction import TextExtractor
from text_extraction.text_extraction_noPreprocess import TextExtractor
import cv2
import os
from preprocessing.png_proc import preprocessing_general,process_image


class SchematicAnalyzer:
    def __init__(self, model_path,text_detection_enabled=True, preprocess_enabled=True):
        self.text_detection_enabled = text_detection_enabled
        self.preprocess_enabled = preprocess_enabled
        self.block_detector = BlockDetector(model_path=model_path)


    def analyze(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        if self.preprocess_enabled:
            # Preprocess the image
            preprocessed_image = process_image(image)
        else:
            processed_image = image
        text_results = {}
        # Wykrywanie tekstu
        if self.text_detection_enabled:
            # Inicjalizacja ekstraktora tekstu
            text_extractor = TextExtractor(languages=['pl', 'en'], min_confidence=0.2)
            # Wykrywanie tekstu
            text_results = text_extractor.extract_text(image_path=image_path)

        
        # Detect blocks in the image
        boxes,nodes_exist = self.block_detector.detect_electrical_symbols(image_path=image_path, conf_threshold=0.25)

        results = self.associate_text_with_blocks(text_results["blocks"], boxes)
        print(results)
        # Initialize the net detector with the detected blocks
        net_detector = NetDetector(image, boxes,text_results["blocks"], build_nodes= not nodes_exist)

        # Detect connections in the image
        connections = net_detector.find_connections()
        net_detector.visualize_connections()
        # Get the detected blocks and connections
        
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
            
            class_name = self.block_detector.model.names[int(block.cls)]
            associated_results[i] = {
                'block': class_name,
                'coordinates': {
                    'x1': block_x1,
                    'y1': block_y1,
                    'x2': block_x2,
                    'y2': block_y2
                },
                'texts': closest_texts[:1][0]  # Take the closest text only
            }
            
            # # Debug output
            # if block_texts:
            #     component_type = block.cls if hasattr(block, 'cls') else "Unknown"
            #     text_content = [text['text'] for text in block_texts]
            #     print(f"Component {component_type}: Associated with text: {text_content}")

        return associated_results

def main():

    # Ścieżki
    input_image_path = "img/test.png"  # Pojedynczy obraz do analizy schematu
    input_folder = "img"               # Folder z obrazami do detekcji tekstu
    output_preprocessed_folder = "preprocessed"
    output_text_folder = "text_results"
    
    # Przetwarzanie wstępne obrazów
    # if preprocess_enabled:
    #     # Przetwarzamy cały folder z obrazami
    #     preprocessing_general(
    #         input_folder=input_folder,
    #         output_folder=output_preprocessed_folder,
    #         debug=False
    #     )
    #     print(f"Przetworzono obrazy z folderu {input_folder} do {output_preprocessed_folder}")
        
    #     # Aktualizujemy ścieżkę do obrazu do analizy schematu
    #     filename = os.path.basename(input_image_path)
    #     name, ext = os.path.splitext(filename)
    #     processed_image_path = os.path.join(output_preprocessed_folder, f"{name}_processed.png")
    #     print(f"Obraz do analizy schematu: {processed_image_path}")
    # else:
    #     processed_image_path = input_image_path
    
    # # Wykrywanie tekstu
    # if text_detection_enabled:
    #     # Inicjalizacja ekstraktora tekstu
    #     text_extractor = TextExtractor(languages=['pl', 'en'], min_confidence=0.2)
        
    #     # Folder z przetworzonymi obrazami
    #     folder_to_process = output_preprocessed_folder if preprocess_enabled else input_folder
        
    #     # Wykrywanie tekstu na wszystkich obrazach w folderze
    #     text_results = text_extractor.extract_text_from_folder(
    #         input_folder=folder_to_process,
    #         output_folder=output_text_folder,
    #         save_annotated=True,
    #         save_json=True
    #     )
        
    #     # Wyświetl podsumowanie
    #     total_text_blocks = sum(len(result['blocks']) for result in text_results.values())
    #     print(f"Wykryto łącznie {total_text_blocks} fragmentów tekstu w {len(text_results)} obrazach")
    
    # Analiza schematu
    SchematicAnalyzer(model_path="block_detector/models/handwritten.pt").analyze(image_path="img/test copy.png")
    
    #text_extractor = TextExtractor()
    #preprocessor = Preprocessor()
    """
    Wygląd struktury jakiej powinny być zwracane z detektorów
    BlockDetector:
        chyba bezpośrednio results z YOLO, do dalszego użytku przez LineDetector, pozycja w liście będzie odpowiadała id bloku
        
    LineDetector:
        {
            "blocks": [
                {
                    "id": 1,
                    "type": "resistor",
                    "coordinates": {
                        "x1": 0,
                        "y1": 0,
                        "x2": 100,
                        "y2": 100
                    }
                    "connections": [2, 3]
                },
                {
                    "id": 2,
                    "type": "power_supply",
                    "coordinates": {
                        "x1": 0,
                        "y1": 0,
                        "x2": 100,
                        "y2": 100
                    }
                    "connections": [2, 3]
                }
            ]
        }
    TextExtractor:
        Struktura jest podatna na zmiany, ważne żeby były podane wymiary bloków z EasyOCR
        {
            "blocks": [
                {
                    "text": "some text",
                    "coordinates": {
                        "x1": 0,
                        "y1": 0,
                        "x2": 100,
                        "y2": 100
                    }
                },
                {
                    "text": "some text",
                    "coordinates": {
                        "x1": 0,
                        "y1": 0,
                        "x2": 100,
                        "y2": 100
                    }
                }
            ]
        }

    Finalny wygląd JSON-a:
    {
        "blocks": [
            {
                "id": 1,
                "type": "resistor",
                "text": "some text",
                "coordinates": {
                    "x1": 0,
                    "y1": 0,
                    "x2": 100,
                    "y2": 100
                }
                connections (przez id):[2,3]

            },
            {
                "id": 2,
                "type": "power_supply",
                "text": "some text",
                "coordinates": {
                    "x1": 0,
                    "y1": 0,
                    "x2": 100,
                    "y2": 100
                }
                connections (przez id):[1,4]
            }
            ...
        ]
    }
    """
if __name__ == "__main__":
    main()