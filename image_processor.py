import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
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
    
    def preprocess_for_lines(self, image: np.ndarray, blocks=None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Przetwarza obraz do wykrywania linii.
        
        Args:
            image: Oryginalny obraz do przetworzenia
            blocks: Opcjonalna lista bloków, gdzie znaki nie powinny być wykrywane
            
        Returns:
            Tuple zawierające przetworzony obraz i słownik obrazów debugowania
        """
        debug_images = {}
        
        # 1. Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_images['gray'] = gray.copy()
        
        # 1.5 Maskowanie znaków z klawiatury
        masked_chars, char_mask = self._mask_keyboard_characters(gray, blocks)
        debug_images['masked_chars'] = masked_chars.copy()
        
        # Obszary wykryte jako znaki nie powinny być brane pod uwagę przy wykrywaniu linii
        # Tworzymy tymczasową kopię obrazu bez znaków
        no_chars_image = gray.copy()
        # Zastępujemy obszary znaków średnią wartością tła (aby nie wprowadzać nowych krawędzi)
        background_mean = np.mean(gray[char_mask == 0])
        no_chars_image[char_mask > 0] = background_mean
        
        # 2. Zastosuj filtr Gaussa, aby wygładzić szum, ale zachować krawędzie
        blurred = cv2.GaussianBlur(no_chars_image, (3, 3), 0)
        debug_images['blurred'] = blurred.copy()
        
        # 3. Binaryzacja adaptacyjna z mniejszym blokiem i mniejszą stałą
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=8  # Zmniejszona stała, aby wydobyć więcej szczegółów
        )
        # Upewnij się, że obszary znaków są całkowicie czarne (0) w obrazie binarnym
        binary[char_mask > 0] = 0
        debug_images['binary'] = binary.copy()
        
        # 4. Usuwanie szumu - używam mniejszego kernela, aby zachować cienkie linie
        kernel_small = np.ones((1, 1), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        # Ponownie upewnij się, że obszary znaków są wykluczone
        denoised[char_mask > 0] = 0
        debug_images['denoised'] = denoised.copy()
        
        # 5. Wykrywanie krawędzi z niższymi progami
        edges = cv2.Canny(denoised, 20, 80)  # Niższe progi dla wykrywania cienkich linii
        # Wyklucz obszary znaków z wykrytych krawędzi
        edges[char_mask > 0] = 0
        debug_images['edges'] = edges.copy()
        
        # 6. Dylatacja - używam mniejszego kernela dla cienkich linii
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        # Wyklucz obszary znaków z dylatowanych krawędzi
        dilated[char_mask > 0] = 0
        debug_images['dilated'] = dilated.copy()
        
        # 7. Łączenie komponentów
        kernel_connect = np.ones((2, 2), np.uint8)
        connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
        # Wyklucz obszary znaków z połączonych komponentów
        connected[char_mask > 0] = 0
        debug_images['connected'] = connected.copy()
        
        # 8. Dodatkowe wypełnianie dziur w liniach
        filled = connected.copy()
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_connect, iterations=1)
        # Ostateczne wykluczenie obszarów znaków
        filled[char_mask > 0] = 0
        debug_images['filled'] = filled.copy()
        
        return filled, debug_images
    
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

    def _mask_keyboard_characters(self, gray_image: np.ndarray, blocks: List[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa znaki z klawiatury (litery, cyfry i inne znaki) i maskuje je za pomocą małych, nieregularnych kwadratów.
        Wykorzystuje algorytm MSER do wykrywania tekstu.
        Ignoruje obszary bloków, aby nie wykrywać ani nie maskować znaków w ich obrębie.
        
        Args:
            gray_image: Obraz w skali szarości
            blocks: Lista bloków, gdzie każdy blok to słownik z kluczem 'coords' (współrzędne [x1, y1, x2, y2])
            
        Returns:
            Tuple zawierające:
                - Przetworzony obraz z zamaskowanymi znakami
                - Maska wykrytych znaków
        """
        # Kopia obrazu wejściowego
        result = gray_image.copy()
        
        # Przygotowanie maski obszarów, gdzie NIE będziemy szukać znaków (bloki)
        block_mask = np.zeros_like(gray_image)
        
        # Jeśli przekazano bloki, zaznacz ich obszary na masce
        if blocks:
            for block in blocks:
                coords = block['coords']
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                # Dodaj margines do obszaru bloku
                margin = 3
                x1_safe = max(0, x1 - margin)
                y1_safe = max(0, y1 - margin)
                x2_safe = min(gray_image.shape[1] - 1, x2 + margin)
                y2_safe = min(gray_image.shape[0] - 1, y2 + margin)
                # Zaznacz obszar bloku na masce
                cv2.rectangle(block_mask, (x1_safe, y1_safe), (x2_safe, y2_safe), 255, -1)
        
        # Przygotowanie maski na wykryte znaki
        char_mask = np.zeros_like(gray_image)
        
        # 1. Wstępne przetwarzanie - wygładzenie obrazu dla lepszego wykrywania tekstu
        # Redukcja szumu i zwiększenie kontrastu
        blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # 2. Użyj algorytmu MSER do wykrycia regionów tekstu
        # MSER dobrze wykrywa jednolite regiony o stałym kolorze, takie jak tekst
        mser = cv2.MSER_create(
            delta=5,                # Wartość delta dla algorytmu MSER (mniejsza wartość = więcej regionów)
            min_area=10,            # Minimalna powierzchnia regionu
            max_area=5000,          # Maksymalna powierzchnia regionu
            max_variation=0.5,      # Maksymalna wariacja między regionami
            min_diversity=0.3,      # Minimalna różnorodność między regionami
            max_evolution=200,      # Maksymalna liczba kroków ewolucji
            area_threshold=1.01,    # Próg powierzchni dla regionów
            min_margin=0.003,       # Minimalny margines dla regionów
            edge_blur_size=5        # Rozmiar rozmycia krawędzi
        )
        
        # Wykrywanie regionów zarówno na obrazie oryginalnym jak i jego inwersji
        # Pomaga wykryć zarówno jasny tekst na ciemnym tle jak i ciemny tekst na jasnym tle
        regions, _ = mser.detectRegions(blur)
        inverted = 255 - blur
        regions_inv, _ = mser.detectRegions(inverted)
        
        # Połącz regiony z obu detekcji
        all_regions = regions + regions_inv
        
        # 3. Filtracja i grupowanie wykrytych regionów
        text_regions = []
        
        for region in all_regions:
            # Przekształć region na punkty (x,y)
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            x, y, w, h = cv2.boundingRect(hull)
            
            # Sprawdź czy region nie zachodzi na obszar bloku
            # Stwórz maskę regionu
            region_mask = np.zeros_like(gray_image)
            cv2.drawContours(region_mask, [hull], 0, 255, -1)
            
            # Sprawdź nakładanie z blokami - jeśli jakikolwiek piksel regionu nachodzi na blok, pomiń region
            if blocks and np.any(np.logical_and(region_mask > 0, block_mask > 0)):
                continue
            
            # Filtruj regiony według proporcji i rozmiaru typowych dla tekstu
            aspect_ratio = w / float(h) if h > 0 else 0
            area = w * h
            
            # Typowe wartości dla znaków tekstowych
            min_area = 10
            max_area = 3000
            min_aspect = 0.1  # Dla znaków jak 'i' lub '!'
            max_aspect = 4.0  # Dla znaków jak '—' lub '='
            
            # Filtruj według tych kryteriów
            if (min_area <= area <= max_area) and (min_aspect <= aspect_ratio <= max_aspect):
                text_regions.append((x, y, w, h))
        
        # 4. Grupowanie bliskich regionów (np. litery w słowie)
        # Sortuj regiony od lewej do prawej
        text_regions.sort(key=lambda r: r[0])
        
        grouped_regions = []
        current_group = []
        
        for i, region in enumerate(text_regions):
            x, y, w, h = region
            
            if not current_group:
                current_group.append(region)
            else:
                prev_x, prev_y, prev_w, prev_h = current_group[-1]
                
                # Odległość pozioma i pionowa między regionami
                hdist = x - (prev_x + prev_w)
                vdist = abs((y + h/2) - (prev_y + prev_h/2))
                
                # Sprawdź, czy linia łącząca centra regionów nie przecina bloku
                if blocks:
                    center1 = (prev_x + prev_w//2, prev_y + prev_h//2)
                    center2 = (x + w//2, y + h//2)
                    line_mask = np.zeros_like(gray_image)
                    cv2.line(line_mask, center1, center2, 255, 1)
                    # Jeśli linia przecina blok, rozpocznij nową grupę
                    if np.any(np.logical_and(line_mask > 0, block_mask > 0)):
                        if len(current_group) > 0:
                            grouped_regions.append(current_group)
                        current_group = [region]
                        continue
                
                # Jeśli regiony są blisko siebie, dodaj do grupy
                if hdist <= 15 and vdist <= max(h, prev_h)/1.5:
                    current_group.append(region)
                else:
                    # Zakończ grupę i rozpocznij nową
                    if len(current_group) > 0:
                        grouped_regions.append(current_group)
                    current_group = [region]
        
        # Dodaj ostatnią grupę, jeśli istnieje
        if current_group:
            grouped_regions.append(current_group)
        
        # 5. Przetwarzanie zgrupowanych regionów
        for group in grouped_regions:
            # Jeśli grupa zawiera tylko jeden region, sprawdź jego proporcje
            if len(group) == 1:
                x, y, w, h = group[0]
                area = w * h
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Dla pojedynczych znaków stosujemy bardziej restrykcyjne kryteria
                if area < 20 or area > 1000 or aspect_ratio < 0.2 or aspect_ratio > 3.0:
                    continue
            
            # Znajdź prostokąt otaczający całą grupę
            min_x = min(r[0] for r in group)
            min_y = min(r[1] for r in group)
            max_x = max(r[0] + r[2] for r in group)
            max_y = max(r[1] + r[3] for r in group)
            
            # Dodaj margines
            margin = 2
            x1 = max(0, min_x - margin)
            y1 = max(0, min_y - margin)
            x2 = min(gray_image.shape[1] - 1, max_x + margin)
            y2 = min(gray_image.shape[0] - 1, max_y + margin)
            
            # Sprawdź dodatkowo kolizję z blokami
            region_rect_mask = np.zeros_like(gray_image)
            cv2.rectangle(region_rect_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Jeśli jest kolizja z blokiem, pomiń region
            if blocks and np.any(np.logical_and(region_rect_mask > 0, block_mask > 0)):
                continue
            
            # Dziel region na małe kwadraciki zamiast jednego dużego prostokąta
            cell_size = 4  # Rozmiar kwadracików
            for cy in range(y1, y2, cell_size):
                for cx in range(x1, x2, cell_size):
                    # Oblicz granice kwadracika
                    cxe = min(cx + cell_size, x2)
                    cye = min(cy + cell_size, y2)
                    
                    # Upewnij się, że kwadracik nie nachodzi na blok
                    cell_mask = np.zeros_like(gray_image)
                    cv2.rectangle(cell_mask, (cx, cy), (cxe, cye), 255, -1)
                    
                    if blocks and np.any(np.logical_and(cell_mask > 0, block_mask > 0)):
                        continue
                    
                    # Dodaj do maski znaków
                    cv2.rectangle(char_mask, (cx, cy), (cxe, cye), 255, -1)
                    
                    # Zamaskuj wykryte znaki na obrazie wynikowym
                    result[cy:cye, cx:cxe] = 128
            
        return result, char_mask 