import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging

class ImageProcessor:
    """Klasa odpowiedzialna za przetwarzanie obrazu do wykrywania linii."""
    
    def __init__(self):
        """Inicjalizuje procesor obrazu z parametrami."""
        self.params = {
            'canny_threshold1': 10,
            'canny_threshold2': 50,
            'hough_threshold': 10,
            'min_line_gap': 3,
            'min_line_length': 3,
            'max_line_length': 3000,
            'angle_threshold': 75
        }
        self.debug_images = {}
    
    def preprocess_for_lines(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Przetwarza obraz do wykrywania linii."""
        debug_images = {}
        
        # 1. Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_images['gray'] = gray.copy()
        
        # 2. Binaryzacja adaptacyjna
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,  # Zwiększamy rozmiar bloku
            C=15  # Zwiększamy stałą
        )
        debug_images['binary'] = binary.copy()
        
        # 3. Usuwanie szumu
        kernel = np.ones((2, 2), np.uint8)  # Zmniejszamy rozmiar jądra
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # Zmniejszamy liczbę iteracji
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
        debug_images['denoised'] = denoised.copy()
        
        # 4. Wykrywanie krawędzi
        edges = cv2.Canny(denoised, 50, 150)  # Zmniejszamy progi
        debug_images['edges'] = edges.copy()
        
        # 5. Dylatacja
        kernel = np.ones((2, 2), np.uint8)  # Zmniejszamy rozmiar jądra
        dilated = cv2.dilate(edges, kernel, iterations=1)  # Zmniejszamy liczbę iteracji
        debug_images['dilated'] = dilated.copy()
        
        # 6. Łączenie komponentów
        connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Dodajemy operację otwarcia, aby usunąć ciemne podwiaty
        connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel, iterations=1)
        debug_images['connected'] = connected.copy()
        
        return connected, debug_images
    
    def _filter_and_connect_lines(self, image: np.ndarray) -> np.ndarray:
        """Filtruje i łączy linie na obrazie."""
        # Wykryj linie na obrazie
        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, 
            self.params['hough_threshold'],
            minLineLength=self.params['min_line_length'],
            maxLineGap=self.params['min_line_gap']
        )
        
        if lines is None:
            logging.warning("Nie wykryto żadnych linii")
            return np.zeros_like(image)
            
        # Utwórz nowy obraz do rysowania linii
        result = np.zeros_like(image)
        
        # Narysuj wszystkie wykryte linie
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), 255, 2)
            
        logging.info(f"Wykryto i narysowano {len(lines)} linii")
        return result 