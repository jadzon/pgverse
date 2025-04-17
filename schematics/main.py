from block_detector.block_detector import BlockDetector
from net_detector.net_detector import NetDetector
#from text_extraction.text_extraction import TextExtractor
import cv2
import os


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

        # Cut out blocks from the image
        net_detector.cut_out_blocks()
def main():
    
    SchematicAnalyzer(model_path="block_detector/models/handwritten.pt").analyze(image_path="img/test.png")
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