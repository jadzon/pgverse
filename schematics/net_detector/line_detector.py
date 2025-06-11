import os
import json
import cv2
import numpy as np
import shutil
import argparse
from pathlib import Path

def load_json_ignore_regions(json_path, image_shape, debug=False):
    """
    Wczytuje regiony do ignorowania z pliku JSON.
    
    Args:
        json_path: Ścieżka do pliku JSON
        image_shape: Kształt obrazu (wysokość, szerokość)
        debug: Czy wyświetlać informacje debugowania
    
    Returns:
        Lista regionów do ignorowania (wielokąty)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        ignore_regions = []
        
        # Pobierz rzeczywisty rozmiar obrazu
        actual_height, actual_width = image_shape[:2]
        
        # Sprawdzenie czy mamy do czynienia z wymaganym skalowaniem
        need_scaling = False
        
        # Domyślne wartości skalowania
        json_width = actual_width
        json_height = actual_height
        
        # Sprawdź czy JSON zawiera informacje o rozmiarach
        if 'image_sizes' in data:
            # Sprawdź źródło danych (annotations lub text_marked)
            if 'annotations' in data['image_sizes'] and len(data['image_sizes']['annotations']) == 2:
                json_width, json_height = data['image_sizes']['annotations']
                need_scaling = True
                if debug:
                    print(f"Używam rozmiarów z annotations: {json_width}x{json_height}")
            
            # Lub użyj image_size jeśli jest dostępne
            elif 'image_size' in data and 'width' in data['image_size'] and 'height' in data['image_size']:
                json_width = data['image_size']['width']
                json_height = data['image_size']['height']
                need_scaling = True
                if debug:
                    print(f"Używam rozmiarów z image_size: {json_width}x{json_height}")
        
        # Oblicz współczynniki skalowania
        scale_x = actual_width / json_width
        scale_y = actual_height / json_height
        
        if debug:
            print(f"Plik: {json_path}")
            print(f"Rzeczywisty rozmiar obrazu: {actual_width}x{actual_height}")
            print(f"Rozmiar w JSON: {json_width}x{json_height}")
            print(f"Współczynniki skalowania: x={scale_x}, y={scale_y}")
        
        # Traktuj wszystkie bloki jako obszary do ignorowania
        if 'blocks' in data:
            for block in data['blocks']:
                if 'coords' in block and len(block['coords']) == 4:
                    # Format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = block['coords']
                    
                    # Zastosuj skalowanie
                    if need_scaling:
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                    
                    # Konwersja do punktów wielokąta (prostokąt)
                    points = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.int32)
                    
                    ignore_regions.append(points)
        
        return ignore_regions
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku JSON {json_path}: {e}")
        return []

def create_mask_from_ignore_regions(image_shape, ignore_regions):
    """
    Tworzy maskę z regionami do ignorowania.
    
    Args:
        image_shape: Kształt obrazu (wysokość, szerokość)
        ignore_regions: Lista regionów do ignorowania
    
    Returns:
        Maska (biała = obszary do analizy, czarna = obszary do ignorowania)
    """
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255
    
    for region in ignore_regions:
        cv2.fillPoly(mask, [region], 0)  # Czarne wielokąty = obszary do ignorowania
    
    return mask

def is_duplicate_line(line, existing_lines, distance_threshold=10, angle_threshold=0.1):
    """
    Sprawdza, czy linia jest duplikatem istniejących linii.
    
    Args:
        line: Linia do sprawdzenia [x1, y1, x2, y2]
        existing_lines: Lista istniejących linii
        distance_threshold: Próg odległości między liniami
        angle_threshold: Próg kąta między liniami
    
    Returns:
        True jeśli linia jest duplikatem, False w przeciwnym razie
    """
    x1, y1, x2, y2 = line
    
    # Oblicz wektor kierunkowy i długość linii
    vec = np.array([x2 - x1, y2 - y1], dtype=float)
    length = np.linalg.norm(vec)
    
    if length < 1e-6:  # Bardzo krótka linia, traktuj jak punkt
        return True
    
    # Normalizuj wektor kierunkowy
    vec /= length
    
    for ex_line in existing_lines:
        ex_x1, ex_y1, ex_x2, ex_y2 = ex_line
        
        # Oblicz wektor kierunkowy istniejącej linii
        ex_vec = np.array([ex_x2 - ex_x1, ex_y2 - ex_y1], dtype=float)
        ex_length = np.linalg.norm(ex_vec)
        
        if ex_length < 1e-6:
            continue
        
        # Normalizuj wektor
        ex_vec /= ex_length
        
        # Sprawdź, czy linie mają podobny kierunek
        cos_angle = np.abs(np.dot(vec, ex_vec))
        if cos_angle > 1 - angle_threshold:  # Kąt bliski 0 lub 180 stopni
            # Sprawdź, czy linie są blisko siebie
            # Oblicz odległość między punktami końcowymi
            d1 = np.linalg.norm(np.array([x1, y1]) - np.array([ex_x1, ex_y1]))
            d2 = np.linalg.norm(np.array([x2, y2]) - np.array([ex_x2, ex_y2]))
            d3 = np.linalg.norm(np.array([x1, y1]) - np.array([ex_x2, ex_y2]))
            d4 = np.linalg.norm(np.array([x2, y2]) - np.array([ex_x1, ex_y1]))
            
            min_dist = min(d1, d2, d3, d4)
            if min_dist < distance_threshold:
                return True
    
    return False

def filter_lines(lines, min_length=15, duplicate_threshold=10, angle_threshold=0.1, 
                prefer_orthogonal=True, orthogonal_weight=1.5):
    """
    Filtruje linie, aby usunąć duplikaty i preferować linie ortogonalne (pionowe/poziome).
    
    Args:
        lines: Lista linii [[x1, y1, x2, y2], ...]
        min_length: Minimalna długość linii
        duplicate_threshold: Próg odległości dla duplikatów
        angle_threshold: Próg kąta dla duplikatów
        prefer_orthogonal: Czy preferować linie ortogonalne
        orthogonal_weight: Waga dla linii ortogonalnych
    
    Returns:
        Przefiltrowana lista linii
    """
    if lines is None or len(lines) == 0:
        return []
    
    # Wyodrębnij linie z formatu OpenCV i oblicz długości
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Oblicz kąt w radianach, a następnie w stopniach
        angle_rad = np.arctan2(abs(y2 - y1), abs(x2 - x1))
        angle_deg = np.degrees(angle_rad) % 90  # Sprowadź do zakresu 0-90
        
        # Oblicz, jak blisko linia jest do pionu lub poziomu (0 = idealnie ortogonalna)
        orthogonality = min(angle_deg, 90 - angle_deg)
        
        # Przypisz wagę linii
        weight = length
        if prefer_orthogonal:
            # Zwiększ wagę dla linii bliskich pionu/poziomu
            if orthogonality < 5:  # Prawie idealnie ortogonalna (w granicach 5 stopni)
                weight *= orthogonal_weight
        
        if length >= min_length:
            line_data.append((x1, y1, x2, y2, length, weight, orthogonality))
    
    # Posortuj linie według wagi (długość * waga ortogonalności) malejąco
    line_data.sort(key=lambda x: x[5], reverse=True)
    
    # Wybierz tylko unikalne linie
    filtered_lines = []
    for line in line_data:
        x1, y1, x2, y2, _, _, _ = line
        if not is_duplicate_line([x1, y1, x2, y2], filtered_lines, duplicate_threshold, angle_threshold):
            filtered_lines.append([x1, y1, x2, y2])
    
    return filtered_lines

def apply_morphology(edges, kernel_size=3, image_shape=None, edge_density=None):
    """
    Stosuje operacje morfologiczne, aby poprawić wykrywanie krawędzi.
    Dostosowuje podejście dla cienkich i grubych linii.
    
    Args:
        edges: Obraz krawędzi
        kernel_size: Bazowy rozmiar jądra dla operacji morfologicznych
        image_shape: Kształt oryginalnego obrazu
        edge_density: Gęstość krawędzi w obrazie
    
    Returns:
        Obraz krawędzi po operacjach morfologicznych
    """
    # Zachowaj kopię oryginalnych krawędzi
    original_edges = edges.copy()
    
    if edge_density is None or image_shape is None:
        # Użyj domyślnego podejścia, jeśli nie mamy informacji o gęstości krawędzi
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened
    
    # Dla bardzo cienkich linii, prawie nie stosuj operacji morfologicznych
    if edge_density < 0.005:
        # Minimalne przetwarzanie - tylko delikatne zamknięcie
        small_kernel = np.ones((2, 2), np.uint8)
        result = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, small_kernel)
        
        # Sprawdź, ile krawędzi zostało utraconych
        edge_pixels_before = cv2.countNonZero(edges)
        edge_pixels_after = cv2.countNonZero(result)
        
        # Jeśli utracono więcej niż 10% krawędzi, po prostu użyj oryginału
        if edge_pixels_after < edge_pixels_before * 0.9:
            return original_edges
        
        return result
    
    # Dla cienkich linii, używamy tylko zamknięcia
    elif edge_density < 0.01:
        small_kernel = np.ones((2, 2), np.uint8)
        result = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, small_kernel)
        
        # Sprawdź, ile krawędzi zostało utraconych
        edge_pixels_before = cv2.countNonZero(edges)
        edge_pixels_after = cv2.countNonZero(result)
        
        # Jeśli utracono więcej niż 20% krawędzi, użyj oryginału
        if edge_pixels_after < edge_pixels_before * 0.8:
            return original_edges
        
        return result
    
    # Dla średnich linii, użyj zamknięcia i delikatnego otwarcia tylko jeśli nie stracimy zbyt wielu krawędzi
    elif edge_density < 0.02:
        # Najpierw zamknięcie
        close_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
        
        # Sprawdź, ile krawędzi zostało utraconych po zamknięciu
        edge_pixels_before = cv2.countNonZero(edges)
        edge_pixels_after_close = cv2.countNonZero(closed)
        
        # Jeśli utracono więcej niż 30% krawędzi po samym zamknięciu, użyj oryginału
        if edge_pixels_after_close < edge_pixels_before * 0.7:
            return original_edges
        
        # Następnie delikatne otwarcie
        open_kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
        
        # Sprawdź, ile krawędzi zostało utraconych po otwarciu
        edge_pixels_after_open = cv2.countNonZero(opened)
        
        # Jeśli utracono więcej niż 30% krawędzi po otwarciu w porównaniu do zamknięcia, użyj tylko zamknięcia
        if edge_pixels_after_open < edge_pixels_after_close * 0.7:
            return closed
        
        return opened
    
    # Dla grubych linii, możemy zastosować standardowe operacje morfologiczne
    else:
        # Najpierw zamknięcie
        close_kernel = np.ones((2, 2), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
        
        # Następnie otwarcie
        open_kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
        
        # Sprawdź, ile krawędzi zostało utraconych
        edge_pixels_before = cv2.countNonZero(edges)
        edge_pixels_after = cv2.countNonZero(opened)
        
        # Jeśli utracono więcej niż 40% krawędzi, użyj tylko zamknięcia
        if edge_pixels_after < edge_pixels_before * 0.6:
            return closed
        
        return opened

def preprocess_image(image):
    """
    Wstępne przetwarzanie obrazu dla lepszego wykrywania linii.
    
    Args:
        image: Obraz wejściowy
    
    Returns:
        Obraz po wstępnym przetworzeniu
    """
    # Zwiększ kontrast za pomocą CLAHE (Contrast Limited Adaptive Histogram Equalization)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Redukcja szumu za pomocą bilateralnego filtrowania, które zachowuje krawędzie
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return blurred

def enhance_edges(edges, edge_density=None):
    """
    Ulepsza obraz krawędzi dla lepszego wykrywania linii.
    
    Args:
        edges: Obraz krawędzi po operacji Canny
        edge_density: Gęstość krawędzi w obrazie
    
    Returns:
        Ulepszony obraz krawędzi
    """
    # Zachowaj kopię oryginalnego obrazu krawędzi
    original_edges = edges.copy()
    
    # Określ typ obrazu na podstawie gęstości krawędzi
    if edge_density is None:
        edge_pixels = cv2.countNonZero(edges)
        img_area = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / img_area
    
    # Zastosuj różne strategie ulepszania w zależności od gęstości krawędzi
    if edge_density < 0.005:  # Bardzo cienkie linie
        # Spróbuj pogrubić bardzo cienkie linie za pomocą dylatacji
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return dilated
    
    elif edge_density < 0.01:  # Cienkie linie
        # Dla cienkich linii, delikatna dylatacja
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Następnie zastosuj operację szkieletyzacji, aby zachować strukturę linii
        thinned = cv2.ximgproc.thinning(dilated)
        
        # Połącz oryginalny obraz z wycienioną wersją
        enhanced = cv2.bitwise_or(original_edges, thinned)
        return enhanced
    
    elif edge_density < 0.02:  # Średnie linie
        # Dla średnich linii, ulepsz strukturę za pomocą zamknięcia
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Usuń małe artefakty za pomocą filtracji po powierzchni
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        # Stwórz nowy obraz tylko z komponentami o odpowiedniej wielkości
        filtered = np.zeros_like(closed)
        min_size = 10  # Minimalny rozmiar komponentu do zachowania
        
        # Przetwarzaj wszystkie komponenty oprócz tła (indeks 0)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered[labels == i] = 255
        
        return filtered
    
    else:  # Grube linie
        # Dla grubych linii, możemy najpierw wycieniować, a następnie pogrubić
        # aby uzyskać bardziej regularną strukturę
        thinned = cv2.ximgproc.thinning(edges)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thinned, kernel, iterations=1)
        return dilated

def detect_multi_scale_edges(image, mask=None):
    """
    Wykrywa krawędzie na wielu skalach i łączy wyniki.
    
    Args:
        image: Obraz wejściowy
        mask: Maska (opcjonalnie)
    
    Returns:
        Obraz krawędzi wykrytych na wielu skalach
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Zastosuj maskę, jeśli istnieje
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Oblicz średnią jasność obrazu
    non_zero_pixels = gray[gray > 0]
    if len(non_zero_pixels) > 0:
        avg_brightness = np.mean(non_zero_pixels)
    else:
        avg_brightness = 128  # Wartość domyślna
    
    # Utwórz trzy różne obrazy krawędzi z różnymi parametrami
    
    # 1. Detekcja z niskimi progami dla słabszych krawędzi
    low_threshold = int(max(10, avg_brightness * 0.12))
    high_threshold = int(min(150, avg_brightness * 0.4))
    edges1 = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)
    
    # 2. Detekcja ze standardowymi progami
    std_threshold_low = int(max(20, avg_brightness * 0.2))
    std_threshold_high = int(min(200, avg_brightness * 0.6))
    edges2 = cv2.Canny(gray, std_threshold_low, std_threshold_high, apertureSize=3)
    
    # 3. Detekcja z wysokimi progami dla silniejszych krawędzi
    high_threshold_low = int(max(40, avg_brightness * 0.3))
    high_threshold_high = int(min(250, avg_brightness * 0.8))
    edges3 = cv2.Canny(gray, high_threshold_low, high_threshold_high, apertureSize=3)
    
    # Połącz wyniki operacją OR
    combined_edges = cv2.bitwise_or(edges1, edges2)
    combined_edges = cv2.bitwise_or(combined_edges, edges3)
    
    return combined_edges

def verify_line_on_edges(line, edges, min_overlap_ratio=0.5, thickness=1):
    """
    Sprawdza czy linia faktycznie istnieje na obrazie krawędzi.
    
    Args:
        line: Linia do zweryfikowania [x1, y1, x2, y2]
        edges: Obraz krawędzi (binary)
        min_overlap_ratio: Minimalny stosunek pikseli krawędzi pokrywających się z linią
        thickness: Grubość linii do sprawdzenia
        
    Returns:
        True jeśli linia pokrywa się z krawędziami, False w przeciwnym przypadku
    """
    # Utwórz pusty obraz (maska)
    mask = np.zeros_like(edges)
    
    # Narysuj linię na masce
    x1, y1, x2, y2 = line
    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    
    # Oblicz długość linii
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Oblicz liczbę pikseli na narysowanej linii
    line_pixels = cv2.countNonZero(mask)
    
    if line_pixels == 0:  # Zabezpieczenie przed dzieleniem przez zero
        return False
    
    # Znajdź część wspólną maski i obrazu krawędzi
    overlap = cv2.bitwise_and(edges, mask)
    overlap_pixels = cv2.countNonZero(overlap)
    
    # Oblicz stosunek pokrycia
    overlap_ratio = overlap_pixels / line_pixels
    
    # Dostosuj minimalny stosunek pokrycia w zależności od długości linii
    # Dla krótszych linii wymagamy większego pokrycia
    adjusted_min_ratio = min_overlap_ratio
    if line_length < 20:
        adjusted_min_ratio = min_overlap_ratio * 1.3  # Zwiększony próg dla krótkich linii
    elif line_length > 50:
        adjusted_min_ratio = min_overlap_ratio * 0.8
    
    return overlap_ratio >= adjusted_min_ratio

def detect_lines(image_path, json_path, category, debug=False):
    """
    Wykrywa linie na obrazie z pominięciem obszarów do ignorowania.
    Nowe podejście: najpierw wykrywamy krawędzie, potem nakładamy maskę.
    
    Args:
        image_path: Ścieżka do obrazu
        json_path: Ścieżka do pliku JSON z obszarami do ignorowania
        category: Kategoria obrazu (np. Automatyka, Elektroniczne)
        debug: Czy zapisywać pliki debugowania
    
    Returns:
        Oryginalny obraz, obraz z wykrytymi liniami
    """
    try:
        # Wczytaj obraz
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Nie można wczytać obrazu: {image_path}")
            return None, None
        
        # Wczytaj obszary do ignorowania
        ignore_regions = load_json_ignore_regions(json_path, image.shape, debug)
        
        # Utwórz maskę
        mask = create_mask_from_ignore_regions(image.shape, ignore_regions)
        
        # Przygotuj obraz do wykrywania linii - użyj ulepszonego preprocesingu
        preprocessed = preprocess_image(image)
        
        # Pobierz numer schematu z nazwy pliku i utwórz foldery dla debugowania
        schema_name = Path(image_path).stem
        schema_number = ''.join(filter(str.isdigit, schema_name))
        schema_number_int = int(schema_number) if schema_number else 0
        
        # Utwórz folder dla kategorii w debug
        debug_base_dir = Path("results/debug")
        debug_dir = debug_base_dir / category
        schema_dir = None
        
        if debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            # Utwórz folder dla schematu
            schema_dir = debug_dir / f"sch{schema_number_int:03d}"
            schema_dir.mkdir(exist_ok=True)
            
            # Zapisz obraz oryginalny z narysowanymi regionami do ignorowania
            debug_image = image.copy()
            for region in ignore_regions:
                cv2.polylines(debug_image, [region], True, (0, 0, 255), 2)
            cv2.imwrite(str(schema_dir / "01_regions.jpg"), debug_image)
            
            # Zapisz maskę
            cv2.imwrite(str(schema_dir / "02_mask.jpg"), mask)
            cv2.imwrite(str(schema_dir / "03_preprocessed.jpg"), preprocessed)
        
        # NOWE PODEJŚCIE: Najpierw wykryj krawędzie na całym obrazie
        edges = detect_multi_scale_edges(preprocessed, None)
        
        if debug:
            cv2.imwrite(str(schema_dir / "04a_edges_full.jpg"), edges)
        
        # Teraz nałóż maskę na wykryte krawędzie (usuń krawędzie w obszarach ignorowanych)
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Policz gęstość krawędzi po maskowaniu
        edge_pixels_initial = cv2.countNonZero(edges_masked)
        img_area = image.shape[0] * image.shape[1]
        edge_density = edge_pixels_initial / img_area
        
        # Zapisujemy krawędzie po nałożeniu maski
        if debug:
            cv2.imwrite(str(schema_dir / "04b_edges_masked.jpg"), edges_masked)
        
        # Zastosuj operacje morfologiczne do poprawy krawędzi
        edges_morphed = apply_morphology(
            edges_masked, 
            kernel_size=3, 
            image_shape=image.shape, 
            edge_density=edge_density
        )
        
        # Policz liczbę pikseli krawędzi po operacjach morfologicznych
        edge_pixels = cv2.countNonZero(edges_morphed)
        edge_density_after = edge_pixels / img_area
        
        if debug:
            cv2.imwrite(str(schema_dir / "04c_edges_morphed.jpg"), edges_morphed)
        
        # Usuń krawędzie wokół ignorowanych obszarów
        # Utwórz poszerzoną maskę dla obszarów do ignorowania
        dilated_inverse_mask = np.ones_like(mask)
        
        # Dla każdego regionu do ignorowania, narysuj wypełniony wielokąt z marginesem
        for region in ignore_regions:
            # Utwórz tymczasową maskę
            temp_mask = np.zeros_like(mask)
            cv2.fillPoly(temp_mask, [region], 255)
            
            # Rozszerz obszar ignorowany
            kernel = np.ones((5, 5), np.uint8)
            dilated_temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)
            
            # Odejmij od głównej maski
            dilated_inverse_mask = cv2.bitwise_and(dilated_inverse_mask, cv2.bitwise_not(dilated_temp_mask))
        
        # Zastosuj rozszerzoną maskę do krawędzi
        edges_cleaned = cv2.bitwise_and(edges_morphed, edges_morphed, mask=dilated_inverse_mask)
        
        if debug:
            cv2.imwrite(str(schema_dir / "04d_edges_cleaned.jpg"), edges_cleaned)
            # Zapisz również rozszerzoną maskę dla wizualizacji
            cv2.imwrite(str(schema_dir / "04e_dilated_mask.jpg"), dilated_inverse_mask)
        
        # Parametry HoughLinesP dostosowane do wykrywania linii
        img_size_factor = np.sqrt(img_area) / 500  # Współczynnik skalowania w zależności od wielkości obrazu
        
        # Dostosuj parametry HoughLinesP w zależności od gęstości krawędzi i wielkości obrazu
        if edge_density < 0.005:  # Bardzo cienkie linie
            hough_threshold = max(7, int(9 * img_size_factor))
            min_line_length = max(7, int(9 * img_size_factor))
            max_line_gap = max(10, int(15 * img_size_factor))
        elif edge_density < 0.01:  # Cienkie linie
            hough_threshold = max(10, int(12 * img_size_factor))
            min_line_length = max(10, int(12 * img_size_factor))
            max_line_gap = max(8, int(12 * img_size_factor))
        elif edge_density < 0.02:  # Średnie linie
            hough_threshold = max(12, int(15 * img_size_factor))
            min_line_length = max(12, int(15 * img_size_factor))
            max_line_gap = max(6, int(10 * img_size_factor))
        else:  # Grube linie
            hough_threshold = max(15, int(18 * img_size_factor))
            min_line_length = max(15, int(18 * img_size_factor))
            max_line_gap = max(5, int(8 * img_size_factor))
        
        # Wykryj linie na wyczyszczonych krawędziach
        lines = cv2.HoughLinesP(
            edges_cleaned,
            rho=1, 
            theta=np.pi/180, 
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        # Jeśli nie wykryto żadnych linii, zwróć pusty wynik
        if lines is None or len(lines) == 0:
            print(f"Nie wykryto żadnych linii na obrazie {image_path}")
            return image, image.copy()  # Zwróć oryginał jako wynik
        else:
            print(f"Łącznie wykryto {len(lines)} linii przed filtrowaniem")
        
        # Przefiltruj linie, aby usunąć duplikaty i preferować ortogonalne
        filtered_lines = []
        if lines is not None:
            # Parametry filtrowania dopasowane do gęstości krawędzi
            if edge_density < 0.005:  # Bardzo cienkie linie
                duplicate_threshold = 3  # Najniższy próg dla bardzo cienkich linii
                angle_threshold = 0.12  # Zwiększony próg kąta dla lepszego łączenia
            elif edge_density < 0.01:  # Cienkie linie
                duplicate_threshold = 4  # Niższy próg
                angle_threshold = 0.1
            elif edge_density < 0.02:  # Średnie linie
                duplicate_threshold = 6  # Nieco wyższy próg
                angle_threshold = 0.08
            else:  # Grube linie
                duplicate_threshold = 8  # Standardowy próg dla grubszych linii
                angle_threshold = 0.06  # Niższy próg kąta dla większej precyzji
            
            # Skoryguj próg filtrowania w zależności od wielkości obrazu
            duplicate_threshold = max(3, int(duplicate_threshold * img_size_factor))
            
            # Filtrowanie linii
            filtered_lines = filter_lines(
                lines=lines, 
                min_length=min_line_length,
                duplicate_threshold=duplicate_threshold,
                angle_threshold=angle_threshold,
                prefer_orthogonal=True, 
                orthogonal_weight=1.8  # Jeszcze większa waga dla linii ortogonalnych
            )
        
        # Narysuj wykryte linie na kopii oryginalnego obrazu
        result_image = image.copy()
        line_count = 0
        total_line_length = 0
        
        # Rysowanie przefiltrowanych linii
        if filtered_lines:
            line_count = len(filtered_lines)
            for line in filtered_lines:
                x1, y1, x2, y2 = line
                # Oblicz długość linii
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_line_length += length
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Utwórz obraz porównujący krawędzie i wykryte linie
        if debug and schema_dir:
            # Utwórz czysty obraz dla porównania (czarny)
            comparison_image = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
            
            # Przekonwertuj edges_cleaned na obraz kolorowy (BGR)
            edges_color = cv2.cvtColor(edges_cleaned, cv2.COLOR_GRAY2BGR)
            
            # Umieść obrazy obok siebie
            comparison_image[0:image.shape[0], 0:image.shape[1]] = edges_color
            comparison_image[0:image.shape[0], image.shape[1]:] = result_image
            
            # Dodaj tekst z informacjami o liczbie pikseli krawędzi i wykrytych linii
            cv2.putText(comparison_image, f"Pikseli krawędzi: {edge_pixels} (przed: {edge_pixels_initial})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_image, f"Gęstość krawędzi: {edge_density_after:.4f} (przed: {edge_density:.4f})", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_image, f"Wykrytych linii (przed filtrem): {len(lines)}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_image, f"Wykrytych linii (po filtrze): {line_count}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_image, f"Całkowita długość linii: {int(total_line_length)}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_image, f"Parametry Hough: {hough_threshold}, {min_line_length}, {max_line_gap}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison_image, f"Typ linii: {'bardzo cienkie' if edge_density < 0.005 else 'cienkie' if edge_density < 0.01 else 'średnie' if edge_density < 0.02 else 'grube'}", (10, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Zapisz obraz porównujący
            cv2.imwrite(str(schema_dir / "05b_comparison.jpg"), comparison_image)
            
            # Zapisz obraz wynikowy
            cv2.imwrite(str(schema_dir / "06_result.jpg"), result_image)
            
            line_type = 'bardzo cienkie' if edge_density < 0.005 else 'cienkie' if edge_density < 0.01 else 'średnie' if edge_density < 0.02 else 'grube'
            print(f"Wykryto {line_count} linii na obrazie {image_path} (pikseli krawędzi: {edge_pixels}/{edge_pixels_initial}, gęstość: {edge_density:.4f}, typ linii: {line_type})")
        else:
            print(f"Wykryto {line_count} linii na obrazie {image_path}")
            
        return image, result_image
    
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def process_dataset(dataset_dirs, json_dirs, output_dir="results", debug=False):
    """
    Przetwarza wszystkie obrazy z podanych katalogów.
    
    Args:
        dataset_dirs: Lista katalogów z obrazami
        json_dirs: Lista katalogów z plikami JSON
        output_dir: Katalog wyjściowy
        debug: Czy włączyć tryb debugowania
    """
    # Utwórz katalog wyjściowy, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)
    
    # Jeśli debug włączony, wyczyść folder debug
    if debug:
        debug_dir = Path(output_dir) / "debug"
        if debug_dir.exists():
            print(f"Czyszczę folder debugowania: {debug_dir}")
            try:
                # Zamiast usuwać cały folder, usuń tylko jego zawartość
                for item in os.listdir(debug_dir):
                    item_path = debug_dir / item
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                print("Folder debug został wyczyszczony")
            except Exception as e:
                print(f"Błąd podczas czyszczenia folderu debug: {e}")
        
        os.makedirs(debug_dir, exist_ok=True)
    
    for dataset_dir, json_dir in zip(dataset_dirs, json_dirs):
        # Pobierz wszystkie pliki obrazów i posortuj je według numeru schematu
        image_paths = list(Path(dataset_dir).glob("*.jpg")) + list(Path(dataset_dir).glob("*.png"))
        
        # Sortowanie obrazów według numeru schematu
        def extract_schema_number(path):
            # Pobierz numer schematu z nazwy pliku
            schema_name = Path(path).stem
            schema_number = ''.join(filter(str.isdigit, schema_name))
            # Jeśli nie ma numeru, zwróć 0
            return int(schema_number) if schema_number else 0
        
        # Sortuj obrazy według numerów schematów
        image_paths.sort(key=extract_schema_number)
        
        category = Path(dataset_dir).name  # Użyj .name zamiast split dla kompatybilności ze ścieżkami Windows
        
        # Utwórz katalog dla kategorii
        category_dir = Path(output_dir) / category
        os.makedirs(category_dir, exist_ok=True)
        
        print(f"Przetwarzanie kategorii: {category}")
        print(f"Znaleziono {len(image_paths)} obrazów w {dataset_dir}")
        
        for image_path in image_paths:
            # Tworzymy nazwę pliku JSON - musimy dodać prefiks kategorii
            json_filename = f"{category}_{image_path.stem}.json"
            json_path = Path(json_dir) / json_filename
            
            if not json_path.exists():
                print(f"Brak odpowiadającego pliku JSON dla {image_path} (szukano: {json_path})")
                continue
            
            print(f"Przetwarzanie: {image_path}")
            # Wykryj linie
            original, result = detect_lines(image_path, json_path, category, debug)
            
            if original is not None and result is not None:
                # Utwórz ścieżki wyjściowe
                output_path = category_dir / f"{image_path.stem}_lines.jpg"
                
                # Zapisz obraz wynikowy
                cv2.imwrite(str(output_path), result)
                print(f"Zapisano wynik dla {image_path} do {output_path}")

def main():
    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description='Detektor linii na schematach')
    parser.add_argument('--debug', action='store_true', 
                        help='Włącz tryb debugowania (zapisywanie dodatkowych obrazów)')
    args = parser.parse_args()
    
    # Ścieżki do folderów z obrazami
    dataset_dirs = [
        "Dataset/Automatyka",
        "Dataset/Elektroniczne"
    ]
    
    # Ścieżki do folderów z plikami JSON
    json_dirs = [
        "combined_json/Automatyka",
        "combined_json/Elektroniczne"
    ]
    
    # Włącz lub wyłącz debugowanie (zapisywanie dodatkowych obrazów)
    debug_mode = args.debug
    if debug_mode:
        print("Tryb debugowania włączony - obrazy pośrednie będą zapisywane w folderze results/debug")
    
    # Przetwórz obrazy
    process_dataset(dataset_dirs, json_dirs, debug=debug_mode)

if __name__ == "__main__":
    main()
