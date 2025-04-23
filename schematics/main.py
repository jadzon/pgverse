from block_detector.block_detector import BlockDetector
from net_detector.net_detector import NetDetector
#from text_extraction.text_extraction import TextExtractor
from text_extraction.text_extraction_noPreprocess import TextExtractor
import cv2
import os
from preprocessing.png_proc import preprocessing_general


class SchematicAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path

    def analyze(self, image_path):
        # Load the image
        image = cv2.imread(image_path)

        # Initialize the block detector
        block_detector = BlockDetector(model_path=self.model_path)

        # Detect blocks in the image
        boxes,nodes_exist = block_detector.detect_electrical_symbols(image_path=image_path, conf_threshold=0.25)

        # Initialize the net detector with the detected blocks
        net_detector = NetDetector(image, boxes, build_nodes= not nodes_exist)

        # Detect connections in the image
        net_detector.find_connections()
        net_detector.visualize_connections()

def main():
    # Ustawienia
    preprocess_enabled = True
    text_detection_enabled = True
    
    # Ścieżki
    input_image_path = "img/test.png"  # Pojedynczy obraz do analizy schematu
    input_folder = "img"               # Folder z obrazami do detekcji tekstu
    output_preprocessed_folder = "preprocessed"
    output_text_folder = "text_results"
    
    # Przetwarzanie wstępne obrazów
    if preprocess_enabled:
        # Przetwarzamy cały folder z obrazami
        preprocessing_general(
            input_folder=input_folder,
            output_folder=output_preprocessed_folder,
            debug=False
        )
        print(f"Przetworzono obrazy z folderu {input_folder} do {output_preprocessed_folder}")
        
        # Aktualizujemy ścieżkę do obrazu do analizy schematu
        filename = os.path.basename(input_image_path)
        name, ext = os.path.splitext(filename)
        processed_image_path = os.path.join(output_preprocessed_folder, f"{name}_processed.png")
        print(f"Obraz do analizy schematu: {processed_image_path}")
    else:
        processed_image_path = input_image_path
    
    # Wykrywanie tekstu
    if text_detection_enabled:
        # Inicjalizacja ekstraktora tekstu
        text_extractor = TextExtractor(languages=['pl', 'en'], min_confidence=0.2)
        
        # Folder z przetworzonymi obrazami
        folder_to_process = output_preprocessed_folder if preprocess_enabled else input_folder
        
        # Wykrywanie tekstu na wszystkich obrazach w folderze
        text_results = text_extractor.extract_text_from_folder(
            input_folder=folder_to_process,
            output_folder=output_text_folder,
            save_annotated=True,
            save_json=True
        )
        
        # Wyświetl podsumowanie
        total_text_blocks = sum(len(result['blocks']) for result in text_results.values())
        print(f"Wykryto łącznie {total_text_blocks} fragmentów tekstu w {len(text_results)} obrazach")
    
    # Analiza schematu
    SchematicAnalyzer(model_path="block_detector/models/handwritten.pt").analyze(image_path=processed_image_path)
    
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