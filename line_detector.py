import cv2
import numpy as np
from typing import Dict, Tuple

class LineDetector:
    """Klasa odpowiedzialna za wykrywanie linii między blokami."""
    
    def __init__(self):
        """Inicjalizuje detektor linii."""
        self.params = {
            'hough_threshold': 50,
            'min_line_length': 100,
            'max_line_length': 1000,
            'min_line_gap': 20,
            'angle_threshold': 45
        }
    
    def are_blocks_connected(self, image: np.ndarray, start_block: Dict, end_block: Dict) -> bool:
        """Sprawdza czy dwa bloki są połączone linią."""
        # Pobierz współrzędne bloków
        start_coords = start_block['coords']
        end_coords = end_block['coords']
        
        # Oblicz środki bloków
        start_center = (
            (start_coords[0] + start_coords[2]) // 2,
            (start_coords[1] + start_coords[3]) // 2
        )
        end_center = (
            (end_coords[0] + end_coords[2]) // 2,
            (end_coords[1] + end_coords[3]) // 2
        )
        
        # Wykryj linie na obrazie
        lines = cv2.HoughLinesP(
            image, 1, np.pi/180,
            self.params['hough_threshold'],
            minLineLength=self.params['min_line_length'],
            maxLineGap=self.params['min_line_gap']
        )
        
        if lines is None:
            return False
        
        # Sprawdź czy istnieje linia łącząca środki bloków
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Sprawdź czy linia jest w odpowiednim zakresie długości
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < self.params['min_line_length'] or length > self.params['max_line_length']:
                continue
            
            # Sprawdź czy linia przechodzi blisko środków obu bloków
            if self._is_line_near_point((x1, y1), (x2, y2), start_center) and \
               self._is_line_near_point((x1, y1), (x2, y2), end_center):
                return True
        
        return False
    
    def _is_line_near_point(self, line_start: Tuple[int, int], 
                          line_end: Tuple[int, int], 
                          point: Tuple[int, int], 
                          threshold: int = 20) -> bool:
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