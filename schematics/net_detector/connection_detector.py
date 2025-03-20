import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import os
from scipy.spatial import cKDTree
from models import Connection
from image_processor import ImageProcessor
from geometry import GeometryHelper
from block_detector import BlockGridHelper
from debug_utils import DebugUtils
import logging
from line_detector import LineDetector

class ConnectionDetector:
    """Główna klasa koordynująca wykrywanie połączeń na schemacie."""
    
    def __init__(self):
        """Inicjalizuje detektor połączeń."""
        self.image_processor = ImageProcessor()
        self.geometry_helper = GeometryHelper()
        self.block_grid = BlockGridHelper()
        self.line_detector = LineDetector()
        self.debug_images = {}
    
    def preprocess_for_lines(self, image: np.ndarray, blocks=None) -> np.ndarray:
        """Przetwarza obraz do wykrywania linii."""
        processed, debug_images = self.image_processor.preprocess_for_lines(image, blocks)
        # Dodaj obrazy debugowe do głównej kolekcji
        logging.info(f"Otrzymano {len(debug_images)} obrazów debugowych z ImageProcessor")
        logging.info(f"Lista etapów debugowania: {list(debug_images.keys())}")
        self.debug_images.update(debug_images)
        logging.info(f"Zaktualizowano debug_images w ConnectionDetector. Liczba obrazów: {len(self.debug_images)}")
        return processed
    
    def detect_connections(self, processed_image: np.ndarray, blocks: List[Dict]) -> List[Connection]:
        """Wykrywa połączenia między blokami na obrazie."""
        connections = []
        
        # Pobierz oryginalny obraz z debug_images
        original_image = self.debug_images['original'].copy()
        
        # Stwórz maskę blokującą obszary bloków (wszystkie piksele = 255, bloki = 0)
        block_mask = np.ones_like(processed_image) * 255
        
        # Rysuj bloki na obrazie i stwórz maskę blokującą obszary bloków
        for block in blocks:
            coords = block['coords']
            # Sprawdź format współrzędnych i narysuj prostokąt
            if isinstance(coords, list) and len(coords) >= 4:
                # Rysuj prostokąt bloku na oryginalnym obrazie
                cv2.rectangle(original_image, 
                            (int(float(coords[0])), int(float(coords[1]))), 
                            (int(float(coords[2])), int(float(coords[3]))), 
                            (0, 255, 0), 2)  # Zielony kolor dla bloków
                
                # Dodaj tekst z numerem bloku
                cv2.putText(original_image, 
                           str(blocks.index(block)), 
                           (int(float(coords[0])), int(float(coords[1])) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (0, 255, 0), 
                           2)
                
                # Zaznacz obszar bloku jako 0 na masce (aby ignorować linie wewnątrz bloków)
                x1, y1 = int(float(coords[0])), int(float(coords[1]))
                x2, y2 = int(float(coords[2])), int(float(coords[3]))
                cv2.rectangle(block_mask, (x1, y1), (x2, y2), 0, -1)  # Wypełnij obszar bloku czarnym kolorem
        
        # Zastosuj maskę do przetworzonego obrazu, aby wykluczyć obszary bloków
        masked_image = cv2.bitwise_and(processed_image, block_mask)
        
        # Zapisz obraz z maską do debugowania
        self.debug_images['masked_image'] = masked_image
        
        # Wykryj linie na przetworzonym obrazie z maską
        lines = cv2.HoughLinesP(
            masked_image, 1, np.pi/180, 
            threshold=15,  # Obniżony próg dla większej czułości
            minLineLength=10,  # Zmniejszona minimalna długość linii
            maxLineGap=30  # Zwiększona maksymalna przerwa w linii
        )
        
        # Dodaj obraz z wykrytymi liniami do debug_images
        detected_lines_image = np.zeros_like(processed_image)
        if lines is not None:
            logging.info(f"Wykryto {len(lines)} linii poza blokami")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(detected_lines_image, (x1, y1), (x2, y2), 255, 2)
                # Narysuj linie na oryginalnym obrazie
                cv2.line(original_image, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Pomarańczowy kolor dla linii
        else:
            logging.info("Nie wykryto żadnych linii poza blokami")
        
        self.debug_images['detected_lines'] = detected_lines_image
        
        # Zapisz finalną wersję z narysowanymi liniami i blokami do debug_images
        self.debug_images['final'] = original_image
        
        return connections
    
    def _is_line_near_point(self, line_start: tuple, line_end: tuple, point: tuple, threshold: int = 20) -> bool:
        """Sprawdza czy linia przechodzi blisko danego punktu."""
        x1, y1 = line_start
        x2, y2 = line_end
        px, py = point
        
        # Oblicz odległość punktu od linii
        if x2 - x1 == 0:  # linia pionowa
            distance = abs(px - x1)
        else:
            # Oblicz odległość punktu od linii
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            distance = abs(m * px - py + b) / np.sqrt(1 + m**2)
        
        return distance <= threshold
    
    def _is_valid_connection(self, start: tuple, end: tuple) -> bool:
        """Sprawdza czy połączenie jest prawidłowe."""
        x1, y1 = start
        x2, y2 = end
        
        # Sprawdź długość linii
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < self.image_processor.params['min_line_length'] or \
           length > self.image_processor.params['max_line_length']:
            return False
            
        # Sprawdź kąt linii
        if x2 - x1 != 0:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            if angle > self.image_processor.params['angle_threshold']:
                return False
        
        return True
    
    def _detect_arrow_direction(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple:
        """Wykrywa kierunek strzałki na linii."""
        # Oblicz środek linii
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # Sprawdź obszar wokół środka linii
        roi_size = 20
        roi = image[max(0, mid_y - roi_size):min(image.shape[0], mid_y + roi_size),
                   max(0, mid_x - roi_size):min(image.shape[1], mid_x + roi_size)]
        
        # Sprawdź czy jest strzałka
        if np.any(roi > 0):
            # Oblicz kierunek linii
            dx = x2 - x1
            dy = y2 - y1
            
            # Określ kierunek strzałki
            if abs(dx) > abs(dy):
                # Linia pozioma
                if dx > 0:
                    return True, "right"
                else:
                    return True, "left"
            else:
                # Linia pionowa
                if dy > 0:
                    return True, "down"
                else:
                    return True, "up"
        
        return False, None
    
    def save_debug_images(self, output_dir: str, filename: str):
        """Zapisuje obrazy debugowe."""
        DebugUtils.save_debug_images(self.debug_images, output_dir, filename) 