import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import logging
import math

@dataclass
class Connection:
    """Klasa reprezentująca połączenie między blokami."""
    start_block: int  # indeks bloku początkowego
    end_block: int    # indeks bloku końcowego
    start_point: Tuple[int, int]  # punkt początkowy połączenia
    end_point: Tuple[int, int]    # punkt końcowy połączenia
    is_directed: bool  # czy połączenie jest kierunkowe
    direction: Optional[str] = None  # kierunek strzałki (np. "up", "down", "left", "right")
    connection_type: Optional[str] = "solid"  # typ połączenia (np. "solid", "dashed", "dotted")

class LineDetector:
    """Klasa odpowiedzialna za wykrywanie linii między blokami."""
    
    def __init__(self):
        """Inicjalizuje detektor linii z parametrami."""
        self.params = {
            # Parametry dla przetwarzania obrazu
            'blur_kernel_size': 5,
            'clahe_clip_limit': 3.5,  # Zwiększona wartość dla lepszego kontrastu
            'clahe_grid_size': (8, 8),
            
            # Parametry dla detekcji krawędzi (Canny)
            'canny_threshold1': 15,  # Obniżony dla wykrywania większej liczby krawędzi
            'canny_threshold2': 75,  # Obniżony dla wykrywania większej liczby krawędzi
            
            # Parametry dla transformacji Hougha
            'hough_rho': 1,
            'hough_theta': np.pi/180,
            'hough_threshold': 10,  # Znacznie obniżony dla wykrywania większej liczby linii
            'hough_min_line_length': 10,  # Obniżony dla wykrywania krótszych linii
            'hough_max_line_gap': 30,  # Zwiększony dla lepszego łączenia segmentów
            
            # Parametry dla filtrowania linii
            'min_line_length': 5,  # Obniżony dla akceptowania bardzo krótkich linii
            'max_line_length': 5000,  # Zwiększony dla akceptowania dłuższych linii
            'distance_threshold': 20,  # Dystans dla grupowania podobnych linii
            'angle_threshold': 7,  # Zwiększony zakres kątów dla grupowania linii
            
            # Parametry dla rysowania
            'line_thickness': 2,
            'use_raw_lsd_lines': True,  # Używaj surowych linii LSD w wyniku końcowym
            
            # Dodatkowe parametry
            'use_lsd_detector': True,  # Użyj dodatkowo detektor LSD
            'lsd_only': True,  # Użyj tylko detektor LSD (bez Hough)
            'lsd_weight': 2.0,  # Waga dla linii LSD (gdy używamy obu detektorów)
            'merge_lines': True,  # Czy łączyć podobne linie
            'extend_lines': True,  # Czy wydłużać linie do bloków
            'extension_factor': 0.15,  # Zwiększony współczynnik wydłużania linii
            'intersect_margin': 5,  # Margines dla wykrywania przecięć
            'remove_duplicates': True,  # Usuwanie duplikatów linii
            'use_multiple_thresholds': True,  # Użyj wielu progów dla detekcji Canny
            'save_debug_images': True,  # Czy zapisywać obrazy etapów pośrednich
            'filter_block_outlines': True,  # Czy usuwać linie tworzące obramowanie bloków
            'block_outline_threshold': 10,  # Próg odległości dla identyfikacji linii obramowania (piksele)
            'block_outline_length_ratio': 0.8,  # Minimalna proporcja długości linii w stosunku do boku bloku, aby uznać ją za obramowanie
            'filter_disconnected_lines': False,  # Zmienione na False, aby przywrócić poprzednią jakość połączeń
            'connection_threshold': 15,  # Przywrócony niższy próg
            'min_line_length_to_filter': 30,  # Minimalna długość linii, która będzie filtrowana
            'require_both_ends_connected': False,  # Czy wymagać, aby oba końce linii były połączone z blokami
            'keep_long_lines': True,  # Czy zachować długie linie niezależnie od połączenia z blokami
            'long_line_threshold': 150,  # Długość linii, powyżej której jest ona uznawana za "długą"
        }
        self.result_image = None
        self.debug_images = {}  # Słownik przechowujący obrazy z poszczególnych etapów
        self.debug_dir = None  # Katalog do zapisywania obrazów debugowania
        self.lsd_lines = []  # Lista linii wykrytych przez LSD
        self.filtered_lsd_lines = []  # Lista linii LSD po usunięciu obramowań bloków
        logging.info("LineDetector zainicjalizowany z priorytetem dla LSD")
    
    def detect_lines(self, image: np.ndarray, blocks: List[Dict], debug_dir: str = None) -> List[Connection]:
        """
        Wykrywa linie i połączenia między blokami na obrazie.
        
        Args:
            image: Oryginalny obraz kolorowy
            blocks: Lista bloków, gdzie każdy blok to słownik z kluczem 'coords' (współrzędne [x1, y1, x2, y2])
            debug_dir: Katalog do zapisywania obrazów debugowania
            
        Returns:
            Lista wykrytych połączeń
        """
        try:
            self.debug_dir = debug_dir
            self.debug_images = {}  # Reset debug images
            
            logging.info(f"Rozpoczynam wykrywanie linii na obrazie o rozmiarze {image.shape}, z {len(blocks)} blokami")
            
            # Skopiuj oryginalny obraz do rysowania wyników
            result_image = image.copy()
            
            # Zapisz oryginalny obraz
            if self.debug_dir:
                self._save_debug_image(image, "01_original_image")
            
            # 1. Wstępne przetwarzanie obrazu
            logging.info("Krok 1: Wstępne przetwarzanie obrazu")
            preprocessed = self._preprocess_image(image)
            
            # Zapisz obraz po wstępnym przetwarzaniu
            if self.debug_dir:
                self._save_debug_image(preprocessed, "02_preprocessed")
            
            # 2. Detekcja krawędzi - wielopoziomowa
            logging.info("Krok 2: Detekcja krawędzi")
            all_edges = []
            
            # Standardowa detekcja Canny
            edges_canny = self._detect_edges(preprocessed)
            all_edges.append(edges_canny)
            
            # Zapisz obraz po detekcji krawędzi (standardowej)
            if self.debug_dir:
                self._save_debug_image(edges_canny, "03a_edges_standard")
            
            # Jeśli używamy wielu progów, dodaj dodatkowe detekcje Canny z różnymi progami
            if self.params['use_multiple_thresholds']:
                # Niskie progi dla słabych krawędzi
                low_edges = cv2.Canny(preprocessed, self.params['canny_threshold1'] - 5, 
                                     self.params['canny_threshold2'] - 25)
                all_edges.append(low_edges)
                
                # Zapisz obraz po detekcji krawędzi z niskimi progami
                if self.debug_dir:
                    self._save_debug_image(low_edges, "03b_edges_low_threshold")
                
                # Wysokie progi dla wyraźnych krawędzi
                high_edges = cv2.Canny(preprocessed, self.params['canny_threshold1'] + 10, 
                                      self.params['canny_threshold2'] + 50)
                all_edges.append(high_edges)
                
                # Zapisz obraz po detekcji krawędzi z wysokimi progami
                if self.debug_dir:
                    self._save_debug_image(high_edges, "03c_edges_high_threshold")
            
            # Łączenie wszystkich detekcji krawędzi
            combined_edges = np.zeros_like(edges_canny)
            for edge in all_edges:
                combined_edges = cv2.bitwise_or(combined_edges, edge)
            
            # Zapisz obraz po połączeniu wszystkich detekcji krawędzi
            if self.debug_dir:
                self._save_debug_image(combined_edges, "03d_edges_combined")
            
            # 3. Stwórz maskę wykluczającą obszary bloków
            logging.info("Krok 3: Tworzenie maski bloków")
            mask = self._create_block_mask(image.shape[:2], blocks)
            
            # Zapisz maskę bloków
            if self.debug_dir:
                self._save_debug_image(mask, "04_block_mask")
            
            # 4. Zastosuj maskę do obrazu krawędzi
            logging.info("Krok 4: Nakładanie maski")
            masked_edges = cv2.bitwise_and(combined_edges, mask)
            
            # Zapisz obraz po nałożeniu maski
            if self.debug_dir:
                self._save_debug_image(masked_edges, "05_masked_edges")
            
            # Dodatkowy krok: Morfologiczne przetwarzanie obrazu krawędzi
            kernel = np.ones((3, 3), np.uint8)
            morphed_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Zapisz obraz po morfologicznym przetwarzaniu
            if self.debug_dir:
                self._save_debug_image(morphed_edges, "06_morphed_edges")
            
            # 5. Wykrywanie linii - priorytet dla LSD
            all_lines = []
            
            # Wykryj linie za pomocą LSD (Line Segment Detector)
            logging.info("Krok 5: Wykrywanie linii za pomocą LSD")
            lsd_lines = self._detect_lsd_lines(preprocessed, mask)
            logging.info(f"Wykryto {len(lsd_lines)} linii za pomocą LSD")
            
            # Zapisz linie LSD do późniejszego użycia
            self.lsd_lines = lsd_lines.copy()
                
            # Zapisz obraz z liniami z detektora LSD
            if self.debug_dir:
                lsd_lines_image = image.copy()
                for line in lsd_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(lsd_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                self._save_debug_image(lsd_lines_image, "07_lsd_lines")
                self.debug_images["07_lsd_lines"] = lsd_lines_image
            
            # Filtruj linie LSD, aby usunąć obramowania bloków
            filtered_lsd_lines = self._filter_block_outlines(lsd_lines, blocks)
            logging.info(f"Po usunięciu obramowań bloków zostało {len(filtered_lsd_lines)} linii z {len(lsd_lines)} oryginalnych linii LSD")
            
            # Zachowaj LSD linie bez filtrowania niepołączonych linii
            self.filtered_lsd_lines = filtered_lsd_lines.copy()
            
            # Opcjonalnie filtruj linie LSD jeśli włączony jest filtr dla niepołączonych linii
            if self.params['filter_disconnected_lines']:
                filtered_lsd_lines = self._filter_disconnected_lines(filtered_lsd_lines, blocks)
                logging.info(f"Po usunięciu niepołączonych linii zostało {len(filtered_lsd_lines)} linii")
                # Zaktualizuj filtered_lsd_lines tylko jeśli filtrowanie jest włączone
                self.filtered_lsd_lines = filtered_lsd_lines.copy()
            
            # Zapisz obraz z przefiltrowanymi liniami LSD
            if self.debug_dir:
                filtered_lsd_image = image.copy()
                for line in self.filtered_lsd_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(filtered_lsd_image, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
                self._save_debug_image(filtered_lsd_image, "07b_filtered_lsd_lines")
                self.debug_images["07b_filtered_lsd_lines"] = filtered_lsd_image
                
                # Dodatkowy obraz pokazujący tylko linie połączone z blokami
                connected_lines_image = image.copy()
                for line in self.filtered_lsd_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(connected_lines_image, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
                self._save_debug_image(connected_lines_image, "07c_connected_lines")
                self.debug_images["07c_connected_lines"] = connected_lines_image
            
            # Jeśli używamy tylko LSD, pomijamy transformację Hougha
            if not self.params['lsd_only']:
                # Wykryj linie za pomocą transformacji Hougha
                logging.info("Krok 6: Wykrywanie linii (Transformacja Hougha)")
                hough_lines = self._detect_hough_lines(morphed_edges)
                logging.info(f"Wykryto {len(hough_lines)} linii za pomocą transformacji Hougha")
                
                # Zapisz obraz z liniami z transformacji Hougha
                if self.debug_dir:
                    hough_lines_image = image.copy()
                    for line in hough_lines:
                        x1, y1, x2, y2 = line
                        cv2.line(hough_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
                    self._save_debug_image(hough_lines_image, "08_hough_lines")
                
                # Jeśli mamy priorytetyzować LSD, dodajemy linie LSD kilka razy (waga)
                lsd_weight = int(self.params['lsd_weight'])
                for _ in range(lsd_weight):
                    all_lines.extend(lsd_lines)
                
                # Dodajemy linie Hough tylko raz
                all_lines.extend(hough_lines)
                logging.info(f"Łącznie wykryto {len(all_lines)} linii (z priorytetem dla LSD)")
            else:
                # Używamy tylko linii z LSD
                all_lines = lsd_lines
                logging.info(f"Używamy tylko {len(all_lines)} linii z LSD")
            
            # Zapisz obraz z wszystkimi liniami (po nadaniu wagi)
            if self.debug_dir:
                all_lines_image = image.copy()
                for line in all_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(all_lines_image, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
                self._save_debug_image(all_lines_image, "09_all_lines")
            
            # 8. Łącz i filtruj wykryte linie
            logging.info("Krok 8: Łączenie i filtrowanie linii")
            filtered_lines = self._filter_lines(all_lines)
            logging.info(f"Po filtrowaniu zostało {len(filtered_lines)} linii")
            
            # Zapisz obraz z odfiltrowanymi liniami
            if self.debug_dir:
                filtered_lines_image = image.copy()
                for line in filtered_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(filtered_lines_image, (x1, y1), (x2, y2), (0, 128, 255), 2, cv2.LINE_AA)
                self._save_debug_image(filtered_lines_image, "10_filtered_lines")
            
            # 9. Usuń duplikaty linii (jeśli włączone)
            if self.params['remove_duplicates']:
                logging.info("Krok 9: Usuwanie duplikatów linii")
                filtered_lines = self._remove_duplicate_lines(filtered_lines)
                logging.info(f"Po usunięciu duplikatów zostało {len(filtered_lines)} linii")
                
                # Zapisz obraz z liniami po usunięciu duplikatów
                if self.debug_dir:
                    no_duplicates_image = image.copy()
                    for line in filtered_lines:
                        x1, y1, x2, y2 = line
                        cv2.line(no_duplicates_image, (x1, y1), (x2, y2), (255, 128, 0), 2, cv2.LINE_AA)
                    self._save_debug_image(no_duplicates_image, "11_no_duplicates")
            
            # 10. Wydłuż linie do bloków (jeśli włączone)
            if self.params['extend_lines']:
                logging.info("Krok 10: Wydłużanie linii do bloków")
                extended_lines = self._extend_lines_to_blocks(filtered_lines, blocks)
                logging.info(f"Po wydłużeniu mamy {len(extended_lines)} linii")
                
                # Zapisz obraz z wydłużonymi liniami
                if self.debug_dir:
                    extended_lines_image = image.copy()
                    for line in extended_lines:
                        x1, y1, x2, y2 = line
                        cv2.line(extended_lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
                    self._save_debug_image(extended_lines_image, "12_extended_lines")
            else:
                extended_lines = filtered_lines
            
            # 11. Znajdź połączenia między blokami na podstawie wykrytych linii
            logging.info("Krok 11: Znajdowanie połączeń między blokami")
            connections = self._find_connections(extended_lines, blocks)
            logging.info(f"Znaleziono {len(connections)} połączeń między blokami")
            
            # Obraz z połączeniami zostanie utworzony w _draw_results
            
            # 12. Narysuj wyniki
            logging.info("Krok 12: Rysowanie wyników")
            self._draw_results(result_image, blocks, connections, self.filtered_lsd_lines)
            
            # Zapisz wynikowy obraz z połączeniami
            if self.debug_dir:
                self._save_debug_image(result_image, "13_final_result")
            
            # Zapisz wynikowy obraz do pola klasy
            self.result_image = result_image
            
            logging.info("Zakończono wykrywanie linii")
            return connections
            
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania linii: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Wstępne przetwarzanie obrazu do wykrywania linii.
        
        Args:
            image: Oryginalny obraz kolorowy
            
        Returns:
            Przetworzony obraz w skali szarości
        """
        # Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bilateralny filtr - zachowuje krawędzie lepiej niż Gaussian
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Wygładzanie obrazu
        blurred = cv2.GaussianBlur(bilateral, 
                                  (self.params['blur_kernel_size'], self.params['blur_kernel_size']), 
                                  0)
        
        # Zastosuj CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # aby poprawić kontrast i uwydatnić linie
        clahe = cv2.createCLAHE(clipLimit=self.params['clahe_clip_limit'], 
                               tileGridSize=self.params['clahe_grid_size'])
        enhanced = clahe.apply(blurred)
        
        # Normalizacja histogramu dla poprawy kontrastu
        normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        # Redukcja szumu za pomocą filtra medianowego
        denoised = cv2.medianBlur(normalized, 3)
        
        return denoised
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Wykrywa krawędzie na obrazie za pomocą algorytmu Canny.
        
        Args:
            image: Obraz w skali szarości
            
        Returns:
            Obraz binarny z wykrytymi krawędziami
        """
        # Wykrywanie krawędzi za pomocą algorytmu Canny
        edges = cv2.Canny(image, 
                         self.params['canny_threshold1'], 
                         self.params['canny_threshold2'],
                         apertureSize=3,
                         L2gradient=True)
        
        # Przetwarzanie morfologiczne do poprawy jakości krawędzi
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        return dilated_edges
    
    def _create_block_mask(self, image_shape: Tuple[int, int], blocks: List[Dict]) -> np.ndarray:
        """
        Tworzy maskę, gdzie obszary bloków są wyłączone (czarne).
        Maska opiera się wyłącznie na danych bloków z plików JSON.
        
        Args:
            image_shape: Wymiary obrazu (wysokość, szerokość)
            blocks: Lista bloków do zamaskowania z plików JSON
            
        Returns:
            Maska binarna, gdzie 0 = blok, 255 = tło
        """
        # Inicjuj maskę (cały obraz biały)
        mask = np.ones(image_shape, dtype=np.uint8) * 255
        
        # Zaznacz obszary bloków jako czarne (0)
        # Używamy dokładnie tych współrzędnych, które są w pliku JSON
        for block in blocks:
            coords = block['coords']
            x1, y1 = int(float(coords[0])), int(float(coords[1]))
            x2, y2 = int(float(coords[2])), int(float(coords[3]))
            
            # Rysuj wypełniony prostokąt na masce
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        
        return mask
    
    def _detect_hough_lines(self, edges: np.ndarray) -> List[np.ndarray]:
        """
        Wykrywa linie za pomocą probabilistycznej transformacji Hougha.
        
        Args:
            edges: Obraz binarny z krawędziami
            
        Returns:
            Lista wykrytych linii, gdzie każda linia to [x1, y1, x2, y2]
        """
        # Wykryj linie za pomocą HoughLinesP
        lines = cv2.HoughLinesP(edges,
                               self.params['hough_rho'],
                               self.params['hough_theta'],
                               self.params['hough_threshold'],
                               minLineLength=self.params['hough_min_line_length'],
                               maxLineGap=self.params['hough_max_line_gap'])
        
        if lines is None:
            return []
        
        # Przekształć wynik do listy linii
        return [line[0] for line in lines]
    
    def _detect_lsd_lines(self, image: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """
        Wykrywa linie za pomocą Line Segment Detector (LSD).
        
        Args:
            image: Obraz w skali szarości
            mask: Maska wykluczająca obszary bloków
            
        Returns:
            Lista wykrytych linii w formacie [x1, y1, x2, y2]
        """
        try:
            # Zastosuj maskę do obrazu
            masked_image = cv2.bitwise_and(image, mask)
            
            # Utworzenie detektora LSD
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            
            # Detekcja linii
            lines, width, prec, nfa = lsd.detect(masked_image)
            
            if lines is None:
                return []
            
            # Konwersja do formatu [x1, y1, x2, y2]
            result_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                result_lines.append([int(x1), int(y1), int(x2), int(y2)])
            
            return result_lines
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania linii LSD: {str(e)}")
            return []
    
    def _filter_lines(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filtruje i łączy podobne linie.
        
        Args:
            lines: Lista wykrytych linii
            
        Returns:
            Lista odfiltrowanych i połączonych linii
        """
        if not lines:
            return []
            
        # Filtruj linie o skrajnych długościach
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if self.params['min_line_length'] <= length <= self.params['max_line_length']:
                filtered_lines.append(line)
        
        # Jeśli nie jest włączone łączenie linii, zwróć tylko odfiltrowane
        if not self.params['merge_lines']:
            return filtered_lines
        
        # Grupuj podobne linie
        merged_lines = self._merge_similar_lines(filtered_lines)
        
        return merged_lines
    
    def _remove_duplicate_lines(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """
        Usuwa duplikaty linii, które są bardzo podobne.
        
        Args:
            lines: Lista linii
            
        Returns:
            Lista linii bez duplikatów
        """
        if not lines:
            return []
        
        # Dystans poniżej którego linie uznajemy za duplikaty
        duplicate_threshold = 10
        angle_threshold = 5  # w stopniach
        
        unique_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            dx1, dy1 = x2 - x1, y2 - y1
            length1 = np.sqrt(dx1*dx1 + dy1*dy1)
            angle1 = np.arctan2(dy1, dx1) * 180 / np.pi
            
            # Sprawdź czy linia nie jest duplikatem istniejącej
            is_duplicate = False
            for unique_line in unique_lines:
                x3, y3, x4, y4 = unique_line
                dx2, dy2 = x4 - x3, y4 - y3
                angle2 = np.arctan2(dy2, dx2) * 180 / np.pi
                
                # Sprawdź podobieństwo kątów
                angle_diff = abs((angle1 - angle2 + 180) % 360 - 180)
                if angle_diff > angle_threshold:
                    continue
                
                # Sprawdź odległość między końcami linii
                dist1 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
                dist2 = np.sqrt((x2 - x4)**2 + (y2 - y4)**2)
                dist3 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
                dist4 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
                
                min_dist = min(dist1 + dist2, dist3 + dist4)
                if min_dist < duplicate_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        return unique_lines
    
    def _merge_similar_lines(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """
        Łączy podobne linie w jedną.
        
        Args:
            lines: Lista linii do połączenia
            
        Returns:
            Lista połączonych linii
        """
        if not lines:
            return []
        
        # Konwertuj linie do formatu [x1, y1, x2, y2, angle, length]
        processed_lines = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            
            # Oblicz kąt i długość linii
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            # Normalizacja kąta do zakresu [0, 180)
            angle = (np.arctan2(dy, dx) * 180 / np.pi) % 180
            
            # Dodaj linię z jej kątem i długością
            processed_lines.append([x1, y1, x2, y2, angle, length])
        
        # Posortuj linie według kąta
        processed_lines.sort(key=lambda x: x[4])
        
        # Grupuj linie o podobnym kącie
        angle_threshold = self.params['angle_threshold']  # Maksymalna różnica kątów w stopniach
        distance_threshold = self.params['distance_threshold']  # Maksymalna odległość między liniami
        
        merged = []
        i = 0
        while i < len(processed_lines):
            current_group = [processed_lines[i]]
            current_angle = processed_lines[i][4]
            
            # Znajdź wszystkie linie o podobnym kącie
            j = i + 1
            while j < len(processed_lines) and (processed_lines[j][4] - current_angle) % 180 < angle_threshold:
                current_group.append(processed_lines[j])
                j += 1
            
            # Jeśli znaleziono tylko jedną linię, dodaj ją do wynikowej listy
            if len(current_group) == 1:
                merged.append(current_group[0][:4])  # Dodaj tylko [x1, y1, x2, y2]
                i = j
                continue
            
            # Podziel linie na klastry według odległości
            clusters = []
            for line in current_group:
                x1, y1, x2, y2 = line[:4]
                
                # Sprawdź czy linia należy do istniejącego klastra
                added_to_cluster = False
                for cluster in clusters:
                    # Sprawdź odległość do każdej linii w klastrze
                    for cl_line in cluster:
                        cl_x1, cl_y1, cl_x2, cl_y2 = cl_line[:4]
                        
                        # Oblicz odległość między liniami
                        # Odległość punktu od linii
                        def dist_point_to_line(x, y, x1, y1, x2, y2):
                            # Sprawdź czy punkt jest w zakresie linii
                            if x1 == x2 and y1 == y2:  # Punkt zamiast linii
                                return np.sqrt((x - x1)**2 + (y - y1)**2)
                                
                            # Długość linii
                            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            if line_length == 0:
                                return np.sqrt((x - x1)**2 + (y - y1)**2)
                                
                            # Odległość punktu od prostej zawierającej linię
                            dist = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_length
                            
                            # Sprawdź czy projekcja punktu jest na linii
                            t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length * line_length)
                            if t < 0:
                                return np.sqrt((x - x1)**2 + (y - y1)**2)
                            if t > 1:
                                return np.sqrt((x - x2)**2 + (y - y2)**2)
                                
                            return dist
                        
                        # Oblicz odległości od końców linii do drugiej linii
                        d1 = dist_point_to_line(x1, y1, cl_x1, cl_y1, cl_x2, cl_y2)
                        d2 = dist_point_to_line(x2, y2, cl_x1, cl_y1, cl_x2, cl_y2)
                        d3 = dist_point_to_line(cl_x1, cl_y1, x1, y1, x2, y2)
                        d4 = dist_point_to_line(cl_x2, cl_y2, x1, y1, x2, y2)
                        
                        min_dist = min(d1, d2, d3, d4)
                        
                        if min_dist < distance_threshold:
                            cluster.append(line)
                            added_to_cluster = True
                            break
                    
                    if added_to_cluster:
                        break
                
                # Jeśli nie należy do żadnego klastra, utwórz nowy
                if not added_to_cluster:
                    clusters.append([line])
            
            # Połącz linie w każdym klastrze
            for cluster in clusters:
                if len(cluster) == 1:
                    merged.append(cluster[0][:4])  # Dodaj tylko [x1, y1, x2, y2]
                else:
                    # Znajdź wszystkie punkty końcowe
                    points = []
                    for line in cluster:
                        points.append((line[0], line[1]))  # (x1, y1)
                        points.append((line[2], line[3]))  # (x2, y2)
                    
                    # Oblicz linie przechodzące przez najbardziej odległe punkty
                    if points:
                        # Znajdź główną oś (PCA)
                        points_array = np.array(points)
                        mean = np.mean(points_array, axis=0)
                        centered = points_array - mean
                        
                        # Oblicz kowariancję
                        cov = np.cov(centered.T)
                        eigvals, eigvecs = np.linalg.eig(cov)
                        
                        # Główny kierunek to wektor własny z największą wartością własną
                        main_direction = eigvecs[:, np.argmax(eigvals)]
                        
                        # Znajdź projekcje punktów na główną oś
                        projections = np.dot(centered, main_direction)
                        
                        # Znajdź punkty z minimalną i maksymalną projekcją
                        min_idx = np.argmin(projections)
                        max_idx = np.argmax(projections)
                        
                        # Linia łącząca te dwa punkty
                        p1 = points_array[min_idx]
                        p2 = points_array[max_idx]
                        
                        merged.append([int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])])
            
            i = j
        
        return merged
    
    def _extend_lines_to_blocks(self, lines: List[np.ndarray], blocks: List[Dict]) -> List[np.ndarray]:
        """
        Wydłuża linie tak, aby łączyły się z blokami.
        
        Args:
            lines: Lista linii
            blocks: Lista bloków
            
        Returns:
            Lista wydłużonych linii
        """
        if not lines:
            return []
        
        extended_lines = []
        extension_factor = self.params['extension_factor']
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Kierunek linii
            dx = x2 - x1
            dy = y2 - y1
            
            # Długość linii
            length = np.sqrt(dx**2 + dy**2)
            
            # Znormalizowany kierunek
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
            else:
                continue  # Pomijamy linie o zerowej długości
            
            # Wydłuż linię w obu kierunkach
            extension = length * extension_factor
            
            # Nowe punkty końcowe
            new_x1 = x1 - dx_norm * extension
            new_y1 = y1 - dy_norm * extension
            new_x2 = x2 + dx_norm * extension
            new_y2 = y2 + dy_norm * extension
            
            extended_lines.append([int(new_x1), int(new_y1), int(new_x2), int(new_y2)])
        
        return extended_lines
    
    def _find_connections(self, lines: List[np.ndarray], blocks: List[Dict]) -> List[Connection]:
        """
        Znajduje połączenia między blokami na podstawie wykrytych linii.
        
        Args:
            lines: Lista wykrytych linii
            blocks: Lista bloków
            
        Returns:
            Lista połączeń między blokami
        """
        connections = []
        
        # Oblicz środki bloków
        block_centers = []
        for block in blocks:
            coords = block['coords']
            x1, y1 = int(float(coords[0])), int(float(coords[1]))
            x2, y2 = int(float(coords[2])), int(float(coords[3]))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            block_centers.append((center_x, center_y))
        
        # Dla każdej linii sprawdź, które bloki przecina lub łączy
        for line in lines:
            x1, y1, x2, y2 = line
            connected_blocks = []
            
            # Sprawdź, które bloki przecina ta linia
            for i, block in enumerate(blocks):
                if self._line_intersects_block(line, block):
                    connected_blocks.append(i)
            
            # Jeśli linia przecina dokładnie dwa bloki, mamy połączenie
            if len(connected_blocks) == 2:
                block1_idx, block2_idx = connected_blocks
                
                # Utwórz połączenie
                connection = Connection(
                    start_block=block1_idx,
                    end_block=block2_idx,
                    start_point=(x1, y1),
                    end_point=(x2, y2),
                    is_directed=False,
                    connection_type="solid"  # Domyślnie linia ciągła
                )
                
                # Sprawdź, czy takie połączenie już istnieje
                is_duplicate = False
                for existing_conn in connections:
                    if (existing_conn.start_block == block1_idx and existing_conn.end_block == block2_idx) or \
                       (existing_conn.start_block == block2_idx and existing_conn.end_block == block1_idx):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    connections.append(connection)
        
        # Dodatkowy krok: przejrzyj linie, które nie tworzą bezpośrednich połączeń, 
        # ale mogą reprezentować ważne ścieżki
        if len(connections) == 0:
            # Jeśli nie znaleziono żadnych połączeń bezpośrednich, spróbuj znaleźć potencjalne połączenia
            for line in lines:
                x1, y1, x2, y2 = line
                potential_connections = []
                
                # Oblicz długość linii
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Pomiń krótkie linie, które są mniej prawdopodobne jako połączenia między blokami
                if line_length < 20:
                    continue
                
                # Sprawdź odległość od bloków
                for i, block in enumerate(blocks):
                    coords = block['coords']
                    bx1, by1 = int(float(coords[0])), int(float(coords[1]))
                    bx2, by2 = int(float(coords[2])), int(float(coords[3]))
                    
                    # Sprawdź czy linia jest blisko bloku (używając punktów końcowych)
                    threshold = 15  # Próg odległości w pikselach
                    
                    # Sprawdź odległość punktu początkowego linii od bloku
                    if self._point_near_block((x1, y1), (bx1, by1, bx2, by2), threshold):
                        potential_connections.append((i, 'start'))
                        
                    # Sprawdź odległość punktu końcowego linii od bloku
                    if self._point_near_block((x2, y2), (bx1, by1, bx2, by2), threshold):
                        potential_connections.append((i, 'end'))
                
                # Jeśli linia jest potencjalnie połączona z dwoma różnymi blokami
                unique_blocks = set([block for block, _ in potential_connections])
                if len(unique_blocks) == 2:
                    block_indices = list(unique_blocks)
                    block1_idx, block2_idx = block_indices
                    
                    # Utwórz połączenie
                    connection = Connection(
                        start_block=block1_idx,
                        end_block=block2_idx,
                        start_point=(x1, y1),
                        end_point=(x2, y2),
                        is_directed=False,
                        connection_type="solid"  # Domyślnie linia ciągła
                    )
                    
                    # Sprawdź, czy takie połączenie już istnieje
                    is_duplicate = False
                    for existing_conn in connections:
                        if (existing_conn.start_block == block1_idx and existing_conn.end_block == block2_idx) or \
                           (existing_conn.start_block == block2_idx and existing_conn.end_block == block1_idx):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        connections.append(connection)
        
        return connections
    
    def _line_intersects_block(self, line: np.ndarray, block: Dict) -> bool:
        """
        Sprawdza, czy linia przecina blok.
        Wykorzystuje dokładnie współrzędne bloków z plików JSON.
        
        Args:
            line: Linia [x1, y1, x2, y2]
            block: Słownik bloku z kluczem 'coords' z pliku JSON
            
        Returns:
            True jeśli linia przecina blok, False w przeciwnym razie
        """
        x1, y1, x2, y2 = line
        coords = block['coords']
        bx1, by1 = int(float(coords[0])), int(float(coords[1]))
        bx2, by2 = int(float(coords[2])), int(float(coords[3]))
        
        # Sprawdź, czy którykolwiek koniec linii jest wewnątrz bloku
        if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2) or \
           (bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
            return True
        
        # Definiuj boki bloku
        sides = [
            ((bx1, by1), (bx2, by1)),  # Górny bok
            ((bx2, by1), (bx2, by2)),  # Prawy bok
            ((bx1, by2), (bx2, by2)),  # Dolny bok
            ((bx1, by1), (bx1, by2))   # Lewy bok
        ]
        
        # Sprawdź, czy linia przecina którykolwiek bok bloku
        line_segment = ((x1, y1), (x2, y2))
        for side in sides:
            if self._line_segments_intersect(line_segment, side):
                return True
        
        # Sprawdź, czy linia przechodzi przez blok
        if (x1 < bx1 and x2 > bx2) or (x1 > bx2 and x2 < bx1):
            if (y1 < by1 and y2 > by2) or (y1 > by2 and y2 < by1):
                return True
        
        return False
    
    def _line_segments_intersect(self, line1: Tuple[Tuple[int, int], Tuple[int, int]], 
                                line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """
        Sprawdza, czy dwa odcinki linii się przecinają.
        
        Args:
            line1: Pierwszy odcinek ((x1, y1), (x2, y2))
            line2: Drugi odcinek ((x3, y3), (x4, y4))
            
        Returns:
            True jeśli odcinki się przecinają, False w przeciwnym razie
        """
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Współliniowe
            return 1 if val > 0 else 2  # Zgodnie lub przeciwnie do ruchu wskazówek zegara
            
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        p1, q1 = line1
        p2, q2 = line2
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # Ogólny przypadek przecięcia
        if o1 != o2 and o3 != o4:
            return True
        
        # Specjalne przypadki
        if o1 == 0 and on_segment(p1, p2, q1): return True
        if o2 == 0 and on_segment(p1, q2, q1): return True
        if o3 == 0 and on_segment(p2, p1, q2): return True
        if o4 == 0 and on_segment(p2, q1, q2): return True
        
        return False
    
    def _draw_results(self, image: np.ndarray, blocks: List[Dict], 
                    connections: List[Connection], lines: List[np.ndarray]) -> None:
        """
        Rysuje wyniki detekcji na obrazie.
        
        Args:
            image: Obraz do rysowania
            blocks: Lista bloków
            connections: Lista połączeń
            lines: Lista wykrytych linii
        """
        # Rysuj bloki
        for i, block in enumerate(blocks):
            coords = block['coords']
            x1, y1 = int(float(coords[0])), int(float(coords[1]))
            x2, y2 = int(float(coords[2])), int(float(coords[3]))
            
            # Narysuj prostokąt bloku
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dodaj numer bloku
            cv2.putText(image, str(i), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Jeśli mamy używać przefiltrowanych linii LSD, rysujemy je bezpośrednio
        if self.params['use_raw_lsd_lines'] and self.filtered_lsd_lines:
            logging.info(f"Rysowanie {len(self.filtered_lsd_lines)} przefiltrowanych linii LSD (bez obramowań bloków)")
            for line in self.filtered_lsd_lines:
                x1, y1, x2, y2 = line
                # Rysuj przefiltrowane linie LSD w kolorze czerwonym
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        elif self.params['use_raw_lsd_lines'] and self.lsd_lines:
            logging.info(f"Rysowanie {len(self.lsd_lines)} surowych linii LSD")
            for line in self.lsd_lines:
                x1, y1, x2, y2 = line
                # Rysuj linie LSD w kolorze czerwonym
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        elif self.debug_images.get("07b_filtered_lsd_lines") is not None:
            # Alternatywne podejście - użyj obrazu debug z przefiltrowanymi liniami LSD
            filtered_lsd_lines_copy = self.debug_images["07b_filtered_lsd_lines"].copy()
            cyan_mask = cv2.inRange(filtered_lsd_lines_copy, (0, 200, 200), (100, 255, 255))
            cyan_lines = cv2.bitwise_and(filtered_lsd_lines_copy, filtered_lsd_lines_copy, mask=cyan_mask)
            alpha = 0.7
            cv2.addWeighted(image, 1, cyan_lines, alpha, 0, image)
        elif self.debug_images.get("07_lsd_lines") is not None:
            # Alternatywne podejście - użyj obrazu debug z liniami LSD
            lsd_lines_copy = self.debug_images["07_lsd_lines"].copy()
            red_mask = cv2.inRange(lsd_lines_copy, (0, 0, 150), (100, 100, 255))
            red_lines = cv2.bitwise_and(lsd_lines_copy, lsd_lines_copy, mask=red_mask)
            alpha = 0.7
            cv2.addWeighted(image, 1, red_lines, alpha, 0, image)
        else:
            # Jeśli brak linii LSD, rysuj dostarczone linie
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        
        # Opcjonalnie można narysować obliczone połączenia między blokami
        if False:  # Zmień na True, aby narysować połączenia między blokami
            for connection in connections:
                # Narysuj linię połączenia
                cv2.line(image, connection.start_point, connection.end_point, (255, 0, 0), 
                        self.params['line_thickness'], cv2.LINE_AA)
                
                # Jeśli połączenie jest kierunkowe, narysuj strzałkę
                if connection.is_directed:
                    self._draw_arrow(image, connection.start_point, connection.end_point, 
                                   (255, 0, 0), self.params['line_thickness'])
    
    def _draw_arrow(self, image: np.ndarray, start_point: Tuple[int, int], 
                  end_point: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
        """
        Rysuje strzałkę na końcu linii.
        
        Args:
            image: Obraz do rysowania
            start_point: Punkt początkowy linii
            end_point: Punkt końcowy linii
            color: Kolor linii (B, G, R)
            thickness: Grubość linii
        """
        # Parametry strzałki
        arrow_size = 15
        angle = np.pi / 6  # 30 stopni
        
        # Oblicz wektor kierunkowy linii
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        # Normalizuj wektor
        length = np.sqrt(dx**2 + dy**2)
        if length < 1:
            return
            
        dx /= length
        dy /= length
        
        # Oblicz punkty strzałki
        p1 = (
            int(end_point[0] - arrow_size * (dx * np.cos(angle) + dy * np.sin(angle))),
            int(end_point[1] - arrow_size * (dy * np.cos(angle) - dx * np.sin(angle)))
        )
        
        p2 = (
            int(end_point[0] - arrow_size * (dx * np.cos(angle) - dy * np.sin(angle))),
            int(end_point[1] - arrow_size * (dy * np.cos(angle) + dx * np.sin(angle)))
        )
        
        # Narysuj strzałkę
        cv2.line(image, end_point, p1, color, thickness, cv2.LINE_AA)
        cv2.line(image, end_point, p2, color, thickness, cv2.LINE_AA)
    
    def save_result_image(self, output_dir: str, filename: str) -> None:
        """
        Zapisuje wynikowy obraz.
        
        Args:
            output_dir: Katalog docelowy
            filename: Nazwa pliku (bez rozszerzenia)
        """
        # Upewnij się, że katalog istnieje
        os.makedirs(output_dir, exist_ok=True)
        
        # Zapisz wynikowy obraz
        if self.result_image is not None:
            output_path = os.path.join(output_dir, f"{filename}_result.png")
            cv2.imwrite(output_path, self.result_image)
            logging.info(f"Zapisano wynikowy obraz: {output_path}")

    def _save_debug_image(self, image: np.ndarray, name: str) -> None:
        """
        Zapisuje obraz debugowania do określonego katalogu.
        
        Args:
            image: Obraz do zapisania
            name: Nazwa pliku (bez rozszerzenia)
        """
        if self.debug_dir is None:
            return
            
        # Upewnij się, że katalog istnieje
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Zapisz obraz
        image_path = os.path.join(self.debug_dir, f"{name}.png")
        
        # Jeśli obraz jest w skali szarości, konwertuj go do BGR dla lepszej wizualizacji
        if len(image.shape) == 2:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_to_save = image.copy()
            
        cv2.imwrite(image_path, image_to_save)
        logging.info(f"Zapisano obraz debugowania: {image_path}")

    def _filter_block_outlines(self, lines: List[np.ndarray], blocks: List[Dict]) -> List[np.ndarray]:
        """
        Usuwa linie, które prawdopodobnie tworzą obramowanie bloków.
        
        Args:
            lines: Lista linii w formacie [x1, y1, x2, y2]
            blocks: Lista bloków, gdzie każdy blok to słownik z kluczem 'coords'
            
        Returns:
            Lista przefiltrowanych linii bez obramowań bloków
        """
        if not self.params['filter_block_outlines']:
            return lines
            
        filtered_lines = []
        threshold = self.params['block_outline_threshold']
        length_ratio = self.params['block_outline_length_ratio']
        
        for line in lines:
            x1, y1, x2, y2 = line
            is_outline = False
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            for block in blocks:
                coords = block['coords']
                bx1, by1 = int(float(coords[0])), int(float(coords[1]))
                bx2, by2 = int(float(coords[2])), int(float(coords[3]))
                
                # Oblicz wymiary bloku
                block_width = bx2 - bx1
                block_height = by2 - by1
                
                # Sprawdź, czy linia jest równoległa do któregoś boku bloku i blisko niego
                
                # Górny bok
                if (abs(y1 - by1) < threshold and abs(y2 - by1) < threshold):
                    # Sprawdź, czy linia jest wystarczająco długa w porównaniu do szerokości bloku
                    x_overlap = min(x2, bx2) - max(x1, bx1)
                    if x_overlap > 0 and x_overlap > length_ratio * block_width:
                        is_outline = True
                        break
                    
                # Dolny bok
                if (abs(y1 - by2) < threshold and abs(y2 - by2) < threshold):
                    # Sprawdź, czy linia jest wystarczająco długa w porównaniu do szerokości bloku
                    x_overlap = min(x2, bx2) - max(x1, bx1)
                    if x_overlap > 0 and x_overlap > length_ratio * block_width:
                        is_outline = True
                        break
                    
                # Lewy bok
                if (abs(x1 - bx1) < threshold and abs(x2 - bx1) < threshold):
                    # Sprawdź, czy linia jest wystarczająco długa w porównaniu do wysokości bloku
                    y_overlap = min(y2, by2) - max(y1, by1)
                    if y_overlap > 0 and y_overlap > length_ratio * block_height:
                        is_outline = True
                        break
                    
                # Prawy bok
                if (abs(x1 - bx2) < threshold and abs(x2 - bx2) < threshold):
                    # Sprawdź, czy linia jest wystarczająco długa w porównaniu do wysokości bloku
                    y_overlap = min(y2, by2) - max(y1, by1)
                    if y_overlap > 0 and y_overlap > length_ratio * block_height:
                        is_outline = True
                        break
                        
                # Dodatkowe sprawdzenie dla linii przekątnych, które mogą być częścią obramowania
                if ((abs(x1 - bx1) < threshold and abs(y1 - by1) < threshold) or
                    (abs(x1 - bx1) < threshold and abs(y1 - by2) < threshold) or
                    (abs(x1 - bx2) < threshold and abs(y1 - by1) < threshold) or
                    (abs(x1 - bx2) < threshold and abs(y1 - by2) < threshold) or
                    (abs(x2 - bx1) < threshold and abs(y2 - by1) < threshold) or
                    (abs(x2 - bx1) < threshold and abs(y2 - by2) < threshold) or
                    (abs(x2 - bx2) < threshold and abs(y2 - by1) < threshold) or
                    (abs(x2 - bx2) < threshold and abs(y2 - by2) < threshold)):
                    # Sprawdź, czy linia jest częścią obramowania narożnika
                    corner_threshold = 2 * threshold
                    if (min(abs(x1 - bx1), abs(x1 - bx2)) < corner_threshold and 
                        min(abs(y1 - by1), abs(y1 - by2)) < corner_threshold and
                        min(abs(x2 - bx1), abs(x2 - bx2)) < corner_threshold and
                        min(abs(y2 - by1), abs(y2 - by2)) < corner_threshold):
                        is_outline = True
                        break
            
            if not is_outline:
                filtered_lines.append(line)
        
        return filtered_lines

    def _filter_disconnected_lines(self, lines: List[np.ndarray], blocks: List[Dict]) -> List[np.ndarray]:
        """
        Usuwa linie, które nie są połączone z żadnym blokiem.
        Udoskonalony algorytm, który uwzględnia długość linii i potencjalne połączenia.
        
        Args:
            lines: Lista linii w formacie [x1, y1, x2, y2]
            blocks: Lista bloków, gdzie każdy blok to słownik z kluczem 'coords'
            
        Returns:
            Lista przefiltrowanych linii, które są połączone z co najmniej jednym blokiem
        """
        if not self.params['filter_disconnected_lines']:
            return lines
            
        filtered_lines = []
        threshold = self.params['connection_threshold']
        min_length_to_filter = self.params['min_line_length_to_filter']
        long_line_threshold = self.params['long_line_threshold']
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Oblicz długość linii
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Zachowaj krótkie linie (są mniej prawdopodobne jako "śmieci")
            if line_length < min_length_to_filter:
                filtered_lines.append(line)
                continue
            
            # Zachowaj bardzo długie linie (zazwyczaj są ważne nawet jeśli nie są bezpośrednio połączone)
            if self.params['keep_long_lines'] and line_length > long_line_threshold:
                filtered_lines.append(line)
                continue
                
            # Sprawdź połączenie z blokami
            start_connected = False
            end_connected = False
            directly_intersects = False
            
            for block in blocks:
                coords = block['coords']
                bx1, by1 = int(float(coords[0])), int(float(coords[1]))
                bx2, by2 = int(float(coords[2])), int(float(coords[3]))
                
                # Sprawdź bezpośrednie przecięcie z blokiem
                if self._line_intersects_block(line, block):
                    directly_intersects = True
                    break
                
                # Sprawdź odległość punktu początkowego linii od bloku
                if self._point_near_block((x1, y1), (bx1, by1, bx2, by2), threshold):
                    start_connected = True
                    
                # Sprawdź odległość punktu końcowego linii od bloku
                if self._point_near_block((x2, y2), (bx1, by1, bx2, by2), threshold):
                    end_connected = True
                
                # Jeśli oba końce są połączone lub linia bezpośrednio przecina blok, nie ma potrzeby sprawdzać dalej
                if (start_connected and end_connected) or directly_intersects:
                    break
            
            # Linia jest zachowana, jeśli bezpośrednio przecina blok
            if directly_intersects:
                filtered_lines.append(line)
                continue
                
            # Jeśli wymagamy połączenia obu końców i warunek jest spełniony
            if self.params['require_both_ends_connected'] and start_connected and end_connected:
                filtered_lines.append(line)
                continue
                
            # Jeśli nie wymagamy połączenia obu końców, wystarczy że jeden koniec jest połączony
            if not self.params['require_both_ends_connected'] and (start_connected or end_connected):
                filtered_lines.append(line)
                continue
                
            # Sprawdź, czy linia jest blisko dwóch różnych bloków (potencjalnie łączy je)
            if self._line_connects_blocks(line, blocks, threshold):
                filtered_lines.append(line)
                continue
        
        return filtered_lines
    
    def _line_connects_blocks(self, line: np.ndarray, blocks: List[Dict], threshold: int) -> bool:
        """
        Sprawdza, czy linia potencjalnie łączy dwa różne bloki.
        
        Args:
            line: Linia w formacie [x1, y1, x2, y2]
            blocks: Lista bloków
            threshold: Próg odległości dla uznania połączenia
            
        Returns:
            True jeśli linia potencjalnie łączy dwa różne bloki, False w przeciwnym razie
        """
        x1, y1, x2, y2 = line
        
        # Dla każdej pary bloków sprawdź, czy linia potencjalnie je łączy
        connected_blocks = []
        
        for i, block in enumerate(blocks):
            coords = block['coords']
            bx1, by1 = int(float(coords[0])), int(float(coords[1]))
            bx2, by2 = int(float(coords[2])), int(float(coords[3]))
            
            # Sprawdź, czy linia jest blisko tego bloku
            if (self._point_near_block((x1, y1), (bx1, by1, bx2, by2), threshold) or
                self._point_near_block((x2, y2), (bx1, by1, bx2, by2), threshold) or
                self._line_intersects_block(line, block)):
                connected_blocks.append(i)
                
                # Jeśli linia jest blisko dwóch różnych bloków, to potencjalnie je łączy
                if len(connected_blocks) >= 2:
                    return True
        
        # Sprawdź, czy linia przechodzi blisko środków bloków
        for i, block1 in enumerate(blocks):
            coords1 = block1['coords']
            bx1_1, by1_1 = int(float(coords1[0])), int(float(coords1[1]))
            bx2_1, by2_1 = int(float(coords1[2])), int(float(coords1[3]))
            center1_x = (bx1_1 + bx2_1) // 2
            center1_y = (by1_1 + by2_1) // 2
            
            for j, block2 in enumerate(blocks):
                if i == j:
                    continue  # Pomijamy ten sam blok
                    
                coords2 = block2['coords']
                bx1_2, by1_2 = int(float(coords2[0])), int(float(coords2[1]))
                bx2_2, by2_2 = int(float(coords2[2])), int(float(coords2[3]))
                center2_x = (bx1_2 + bx2_2) // 2
                center2_y = (by1_2 + by2_2) // 2
                
                # Sprawdź, czy linia przechodzi blisko środków obu bloków
                dist1 = self._distance_point_to_line(center1_x, center1_y, x1, y1, x2, y2)
                dist2 = self._distance_point_to_line(center2_x, center2_y, x1, y1, x2, y2)
                
                if dist1 < threshold*1.5 and dist2 < threshold*1.5:
                    return True
        
        return False
    
    def _distance_point_to_line(self, px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Oblicza odległość punktu od linii.
        
        Args:
            px, py: Współrzędne punktu
            x1, y1, x2, y2: Współrzędne linii
            
        Returns:
            Odległość punktu od linii
        """
        # Sprawdź czy punkt jest w zakresie linii
        if x1 == x2 and y1 == y2:  # Punkt zamiast linii
            return np.sqrt((px - x1)**2 + (py - y1)**2)
            
        # Długość linii
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
            
        # Odległość punktu od prostej zawierającej linię
        dist = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_length
        
        # Sprawdź czy projekcja punktu jest na linii
        t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length * line_length)
        if t < 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        if t > 1:
            return np.sqrt((px - x2)**2 + (py - y2)**2)
            
        return dist

    def _point_near_block(self, point: Tuple[int, int], block_coords: Tuple[int, int, int, int], threshold: int) -> bool:
        """
        Sprawdza, czy punkt jest blisko bloku (w odległości threshold pikseli).
        
        Args:
            point: Współrzędne punktu (x, y)
            block_coords: Współrzędne bloku (x1, y1, x2, y2)
            threshold: Maksymalna odległość uznawana za "blisko"
            
        Returns:
            True jeśli punkt jest blisko bloku, False w przeciwnym razie
        """
        x, y = point
        bx1, by1, bx2, by2 = block_coords
        
        # Sprawdź, czy punkt jest wewnątrz bloku
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return True
            
        # Sprawdź odległość do najbliższego boku bloku
        dx = max(0, max(bx1 - x, x - bx2))
        dy = max(0, max(by1 - y, y - by2))
        
        # Oblicz odległość euklidesową
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance <= threshold