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
        
        print(f"Wszystkie wyniki zostały zapisane w folderze {self.results_folder}")
        return {
            "processed_image": processed_image_path,
            "text_results": text_results,
            "boxes": boxes,
            "connections": connections
        }
    
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
        block_detector = BlockDetector(model_path=self.model_path)
        boxes, nodes_exist = block_detector.detect_electrical_symbols(
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
        results_folder="main_results"
    )
    analyzer.analyze(image_path="img/test copy.png")


if __name__ == "__main__":
    main()