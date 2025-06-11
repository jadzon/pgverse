import cv2
import numpy as np
import os
import easyocr
import json
from imutils.object_detection import non_max_suppression
from pathlib import Path

# Ścieżka do katalogu projektu
PROJECT_DIR = "."  

class TextExtractor:
    def __init__(self, languages=['pl', 'en'], min_confidence=0.2):
        """
        Inicjalizacja ekstraktora tekstu.
        
        Args:
            languages: Lista języków do rozpoznawania (domyślnie polski i angielski)
            min_confidence: Minimalny poziom pewności dla detekcji tekstu
        """
        self.languages = languages
        self.min_confidence = min_confidence
        print(f"Inicjalizacja modelu EasyOCR dla języków: {', '.join(languages)}...")
        self.reader = easyocr.Reader(languages)
        print("Model EasyOCR załadowany")
    
    def extract_text(self, image_path, output_folder=None, save_annotated=False, save_json=False):
        """
        Wykrywa tekst na obrazie i opcjonalnie zapisuje wyniki.
        
        Args:
            image_path: Ścieżka do obrazu
            output_folder: Folder wyjściowy dla wyników (opcjonalny)
            save_annotated: Czy zapisać obraz z oznaczeniami
            save_json: Czy zapisać wyniki w formacie JSON
            
        Returns:
            dict: Słownik z wynikami detekcji tekstu
        """
        # Wczytaj obraz
        image = cv2.imread(image_path)
        if image is None:
            print(f"Nie można wczytać obrazu: {image_path}")
            return None
        
        # Wykryj tekst
        detected_texts = self._detect_text(image)
        
        # Przygotuj dane wyjściowe
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        
        # Określ folder wyjściowy
        if output_folder is None:
            output_folder = os.path.join(PROJECT_DIR, "text_detection_results")
        
        # Utwórz folder wyjściowy jeśli nie istnieje
        os.makedirs(output_folder, exist_ok=True)
        
        # Zapisz obraz z oznaczeniami jeśli wymagane
        if save_annotated:
            annotated_image = self._annotate_detected_text(image, detected_texts)
            annotated_path = os.path.join(output_folder, f"{name}_text_detected.png")
            cv2.imwrite(annotated_path, annotated_image)
            print(f"Zapisano obraz z oznaczeniami: {annotated_path}")
        
        # Przygotuj wyniki w formacje JSON
        height, width = image.shape[:2]
        result_json = {
            "image_path": image_path,
            "blocks": [],
            "image_size": {
                "width": width,
                "height": height
            }
        }
        
        # Dodaj każdy wykryty tekst
        for bbox, text, confidence in detected_texts:
            # Konwertuj bbox z formatu [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # na [x_min, y_min, x_max, y_max]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            block = {
                "coords": [
                    float(x_min),
                    float(y_min),
                    float(x_max),
                    float(y_max)
                ],
                "type": "rectangle",
                "text": text,
                "confidence": float(confidence)
            }
            
            result_json["blocks"].append(block)
        
        # Zapisz wyniki w formacie JSON jeśli wymagane
        if save_json:
            json_path = os.path.join(output_folder, f"{name}_text.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            print(f"Zapisano wyniki w JSON: {json_path}")
        
        return result_json
    
    def extract_text_from_folder(self, input_folder, output_folder=None, save_annotated=True, save_json=True):
        """
        Wykrywa tekst na wszystkich obrazach w folderze.
        
        Args:
            input_folder: Folder z obrazami wejściowymi
            output_folder: Folder wyjściowy dla wyników
            save_annotated: Czy zapisać obrazy z oznaczeniami
            save_json: Czy zapisać wyniki w formacie JSON
            
        Returns:
            dict: Słownik z wynikami detekcji tekstu dla wszystkich obrazów
        """
        # Określ folder wyjściowy
        if output_folder is None:
            output_folder = os.path.join(PROJECT_DIR, "text_detection_results")
        
        # Utwórz folder wyjściowy jeśli nie istnieje
        os.makedirs(output_folder, exist_ok=True)
        
        # Wczytaj wszystkie obrazy z folderu
        images_paths = []
        for ext in [".jpg", ".jpeg", ".png"]:
            images_paths.extend(list(Path(input_folder).glob(f"*{ext}")))
            images_paths.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
        
        # Słownik na wyniki
        all_results = {}
        
        # Przetwórz każdy obraz
        for image_path in images_paths:
            print(f"Przetwarzanie obrazu: {image_path}")
            result = self.extract_text(
                str(image_path), 
                output_folder, 
                save_annotated, 
                save_json
            )
            if result:
                all_results[os.path.basename(image_path)] = result
        
        print(f"Przetworzono {len(all_results)} obrazów")
        return all_results
    
    def _detect_text(self, image):
        """
        Wykrywa tekst na obrazie za pomocą EasyOCR.
        
        Args:
            image: Obraz w formacie OpenCV (numpy array)
            
        Returns:
            Lista wykrytych tekstów z pozycjami i prawdopodobieństwami
        """
        # Konwersja do RGB dla EasyOCR
        if len(image.shape) == 2:  # Jeśli obraz jest w skali szarości
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Wykrywanie tekstu
        results = self.reader.readtext(
            img_rgb,
            detail=1,
            paragraph=False,
            decoder='beamsearch',
            beamWidth=5,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=self.min_confidence,
            link_threshold=0.3,
            mag_ratio=1.5
        )
        
        # Filtrowanie wyników o niskim poziomie pewności
        filtered_results = [result for result in results if result[2] >= self.min_confidence]
        
        return filtered_results
    
    def _annotate_detected_text(self, image, detected_texts):
        """
        Oznacza wykryty tekst na obrazie.
        
        Args:
            image: Oryginalny obraz
            detected_texts: Lista wykrytych tekstów z pozycjami i prawdopodobieństwami
            
        Returns:
            Obraz z oznaczeniami
        """
        annotated_image = image.copy()
        
        for (bbox, text, confidence) in detected_texts:
            # Konwertujemy bounding box do formatu punktów
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            
            # Dobieramy kolor w zależności od pewności (od czerwonego przez żółty do zielonego)
            r = int(255 * (1 - confidence))
            g = int(255 * confidence)
            b = 0
            color = (b, g, r)  # BGR w OpenCV
            
            # Rysujemy prostokąt
            cv2.line(annotated_image, tl, tr, color, 2)
            cv2.line(annotated_image, tr, br, color, 2)
            cv2.line(annotated_image, br, bl, color, 2)
            cv2.line(annotated_image, bl, tl, color, 2)
            
            # Przygotowujemy tekst z pewnością
            text_to_display = f"{text}: {confidence:.2f}"
            
            # Ustawiamy tło pod tekstem
            (text_width, text_height), _ = cv2.getTextSize(
                text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated_image,
                (tl[0], tl[1] - text_height - 10),
                (tl[0] + text_width, tl[1]),
                (0, 0, 0),
                -1
            )
            
            # Umieszczamy tekst nad prostokątem
            cv2.putText(
                annotated_image, 
                text_to_display, 
                (tl[0], tl[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        return annotated_image

def main():
    """
    Przykład użycia ekstraktora tekstu jako samodzielnego modułu.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Ekstrakcja tekstu z obrazów')
    parser.add_argument('--input', type=str, required=True,
                        help='Ścieżka do obrazu lub folderu z obrazami')
    parser.add_argument('--output', type=str, default=None,
                        help='Folder wyjściowy (opcjonalnie)')
    parser.add_argument('--lang', type=str, default='pl',
                        help='Języki do wykrywania, oddzielone przecinkami (np. pl,en)')
    parser.add_argument('--conf', type=float, default=0.2,
                        help='Minimalny poziom pewności (0.0-1.0)')
    parser.add_argument('--no-annotate', action='store_true',
                        help='Nie zapisuj obrazów z oznaczeniami')
    parser.add_argument('--no-json', action='store_true',
                        help='Nie zapisuj wyników w formacie JSON')
    args = parser.parse_args()
    
    # Przygotuj listę języków
    languages = args.lang.split(',')
    
    # Inicjalizuj ekstraktor tekstu
    extractor = TextExtractor(languages=languages, min_confidence=args.conf)
    
    # Sprawdź czy podana ścieżka to plik czy folder
    if os.path.isfile(args.input):
        # Przetwórz pojedynczy obraz
        extractor.extract_text(
            args.input,
            args.output,
            not args.no_annotate,
            not args.no_json
        )
    elif os.path.isdir(args.input):
        # Przetwórz wszystkie obrazy w folderze
        extractor.extract_text_from_folder(
            args.input,
            args.output,
            not args.no_annotate,
            not args.no_json
        )
    else:
        print(f"Podana ścieżka nie istnieje: {args.input}")

if __name__ == "__main__":
    main()
