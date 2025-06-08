import os
import cv2
import numpy as np
import json
from collections import defaultdict
from small_text_ocr import detect_exponent_notation, clean_duplicated_text

def extend_boxes(text_blocks, extension_factor=1.5, min_extension=10):
    """
    Sztucznie rozszerza bounding boxy bloków tekstu, aby zwiększyć szansę 
    wykrycia nakładających się bloków tworzących oś.
    
    Args:
        text_blocks: Lista bloków tekstu w formacie [(bbox, text, confidence), ...]
        extension_factor: Współczynnik rozszerzenia boksu (1.5 = +50%)
        min_extension: Minimalna liczba pikseli do rozszerzenia w każdym kierunku
        
    Returns:
        Lista rozszerzonych boksów w formacie [(id, extended_box, original_box, text, confidence), ...]
    """
    extended_boxes = []
    
    for i, (bbox, text, confidence) in enumerate(text_blocks):
        # Przekształć bbox na prostokąt (x_min, y_min, x_max, y_max)
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        
        # Oblicz szerokość i wysokość
        width = x_max - x_min
        height = y_max - y_min
        
        # Oblicz rozszerzenie
        x_extend = max(min_extension, width * (extension_factor - 1) / 2)
        y_extend = max(min_extension, height * (extension_factor - 1) / 2)
        
        # Rozszerz bbox
        ext_x_min = max(0, x_min - x_extend)
        ext_y_min = max(0, y_min - y_extend)
        ext_x_max = x_max + x_extend
        ext_y_max = y_max + y_extend
        
        # Stwórz rozszerzony bbox w tym samym formacie co oryginalny (4 punkty)
        extended_bbox = [
            [ext_x_min, ext_y_min],
            [ext_x_max, ext_y_min],
            [ext_x_max, ext_y_max],
            [ext_x_min, ext_y_max]
        ]
        
        # Zapisz ID, rozszerzony bbox, oryginalny bbox, tekst i pewność
        extended_boxes.append((i, extended_bbox, bbox, text, confidence))
    
    return extended_boxes

def calculate_overlap(box1, box2):
    """
    Oblicza nakładanie się dwóch prostokątów.
    
    Args:
        box1, box2: Boxy w formacie [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        
    Returns:
        Wartość IoU (Intersection over Union) w zakresie 0-1
    """
    # Konwersja do formatu (x_min, y_min, x_max, y_max)
    def get_box_coordinates(box):
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    box1_coords = get_box_coordinates(box1)
    box2_coords = get_box_coordinates(box2)
    
    # Oblicz przecięcie
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # Brak nakładania
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Oblicz uniż
    box1_area = (box1_coords[2] - box1_coords[0]) * (box1_coords[3] - box1_coords[1])
    box2_area = (box2_coords[2] - box2_coords[0]) * (box2_coords[3] - box2_coords[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    iou = intersection_area / union_area
    return max(0.0, min(iou, 1.0))

def is_aligned(box1, box2, axis='x', tolerance=0.3):
    """
    Sprawdza, czy dwa boxy są wyrównane wzdłuż określonej osi.
    
    Args:
        box1, box2: Boxy w formacie [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        axis: Oś wyrównania ('x' dla poziomego, 'y' dla pionowego)
        tolerance: Tolerancja wyrównania (0-1, gdzie 0 oznacza idealnie wyrównane)
        
    Returns:
        Boolean wskazujący, czy boxy są wyrównane
    """
    # Konwersja do formatu (x_min, y_min, x_max, y_max)
    def get_box_coordinates(box):
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    box1_coords = get_box_coordinates(box1)
    box2_coords = get_box_coordinates(box2)
    
    if axis == 'x':
        # Porównaj środki y
        y1_center = (box1_coords[1] + box1_coords[3]) / 2
        y2_center = (box2_coords[1] + box2_coords[3]) / 2
        
        # Wysokość dla normalizacji
        h1 = box1_coords[3] - box1_coords[1]
        h2 = box2_coords[3] - box2_coords[1]
        avg_height = (h1 + h2) / 2
        
        # Normalizowana różnica
        diff = abs(y1_center - y2_center) / avg_height
        return diff <= tolerance
    else:  # axis == 'y'
        # Porównaj środki x
        x1_center = (box1_coords[0] + box1_coords[2]) / 2
        x2_center = (box2_coords[0] + box2_coords[2]) / 2
        
        # Szerokość dla normalizacji
        w1 = box1_coords[2] - box1_coords[0]
        w2 = box2_coords[2] - box2_coords[0]
        avg_width = (w1 + w2) / 2
        
        # Normalizowana różnica
        diff = abs(x1_center - x2_center) / avg_width
        return diff <= tolerance

def detect_axis_groups(extended_boxes, overlap_threshold=0.01, alignment_tolerance=0.3):
    """
    Wykrywa grupy bloków tekstu, które mogą tworzyć osie wykresu.
    
    Args:
        extended_boxes: Lista rozszerzonych bloków w formacie [(id, extended_box, original_box, text, confidence), ...]
        overlap_threshold: Minimalny próg IoU dla uznania nakładania (0-1)
        alignment_tolerance: Tolerancja wyrównania dla bloków w osi (0-1)
        
    Returns:
        Dict zawierający wykryte osie X i Y z przypisanymi blokami
    """
    print("--- DEBUG: Rozpoczynam detect_axis_groups --- ")
    horizontal_axes = []  # Osie X
    vertical_axes = []    # Osie Y
    
    # Sprawdź wszystkie pary bloków
    n = len(extended_boxes)
    horizontal_connections = defaultdict(set)
    vertical_connections = defaultdict(set)
    
    for i in range(n):
        id_i, ext_box_i, orig_box_i, text_i, conf_i = extended_boxes[i]
        
        for j in range(i + 1, n):
            id_j, ext_box_j, orig_box_j, text_j, conf_j = extended_boxes[j]
            
            # Sprawdź, czy rozszerzone boxy nakładają się
            overlap = calculate_overlap(ext_box_i, ext_box_j)
            print(f"  DEBUG: Sprawdzam parę ID ({id_i}, {id_j}), Tekst: ('{text_i}', '{text_j}'), Rozszerzony overlap = {overlap:.3f}")
            
            if overlap > overlap_threshold:
                print(f"    DEBUG: Para ID ({id_i}, {id_j}): Rozszerzony overlap (>{overlap_threshold:.3f}) OK.")
                aligned_horizontally = is_aligned(orig_box_i, orig_box_j, axis='x', tolerance=alignment_tolerance)
                aligned_vertically = is_aligned(orig_box_i, orig_box_j, axis='y', tolerance=alignment_tolerance)
                print(f"      DEBUG: ID ({id_i}, {id_j}): Wyrównanie oryginalnych boxów -> Poziome: {aligned_horizontally} (tolerancja: {alignment_tolerance}), Pionowe: {aligned_vertically} (tolerancja: {alignment_tolerance})")
                
                if aligned_horizontally:
                    # Możliwa oś X (pozioma)
                    print(f"        DEBUG: ID ({id_i}, {id_j}) -> Połączenie poziome dodane.")
                    horizontal_connections[id_i].add(id_j)
                    horizontal_connections[id_j].add(id_i)
                
                if aligned_vertically:
                    # Możliwa oś Y (pionowa)
                    print(f"        DEBUG: ID ({id_i}, {id_j}) -> Połączenie pionowe dodane.")
                    vertical_connections[id_i].add(id_j)
                    vertical_connections[id_j].add(id_i)
            # else: # Opcjonalny log dla braku overlapu
                # print(f"    DEBUG: Para ID ({id_i}, {id_j}): Brak wystarczającego overlapu rozszerzonych boxów ({overlap:.3f} <= {overlap_threshold:.3f}).")

    print(f"  DEBUG: Liczba potencjalnych połączeń poziomych: {len(horizontal_connections)}")
    print(f"  DEBUG: Liczba potencjalnych połączeń pionowych: {len(vertical_connections)}")
    
    # Znajdź grupy połączonych bloków dla osi X
    visited_h = set()
    for node in horizontal_connections:
        if node not in visited_h:
            axis_group = []
            _find_connected_components(node, horizontal_connections, visited_h, axis_group)
            if len(axis_group) >= 2:  # Co najmniej 2 elementy tworzą oś
                # Konwersja ID na pełne informacje o blokach
                print(f"    DEBUG: Wykryto poziomą grupę osi (X) z {len(axis_group)} elementami: {axis_group}")
                axis_blocks = [extended_boxes[id] for id in axis_group]
                horizontal_axes.append(axis_blocks)
    
    # Znajdź grupy połączonych bloków dla osi Y
    visited_v = set()
    for node in vertical_connections:
        if node not in visited_v:
            axis_group = []
            _find_connected_components(node, vertical_connections, visited_v, axis_group)
            if len(axis_group) >= 2:  # Co najmniej 2 elementy tworzą oś
                # Konwersja ID na pełne informacje o blokach
                print(f"    DEBUG: Wykryto pionową grupę osi (Y) z {len(axis_group)} elementami: {axis_group}")
                axis_blocks = [extended_boxes[id] for id in axis_group]
                vertical_axes.append(axis_blocks)
    
    return {
        'horizontal': horizontal_axes,
        'vertical': vertical_axes
    }

def _find_connected_components(node, connections, visited, group):
    """
    Pomocnicza funkcja DFS do znajdowania połączonych komponentów grafu.
    """
    visited.add(node)
    group.append(node)
    
    for neighbor in connections[node]:
        if neighbor not in visited:
            _find_connected_components(neighbor, connections, visited, group)

def select_best_axis_for_visualization(axes_list):
    """
    Wybiera najlepszą oś do wizualizacji na podstawie jakości elementów.
    
    Args:
        axes_list: Lista osi
        
    Returns:
        Najlepsza oś lub None jeśli lista jest pusta
    """
    if not axes_list:
        return None
    
    def axis_quality_score(axis):
        """Oblicza jakość osi na podstawie zawartości tekstowej"""
        score = 0
        valid_elements = 0
        
        for _, _, _, text, confidence in axis:
            # Sprawdź czy element zawiera tekst liczbowy lub potęgowy
            text_clean = text.strip()
            
            # Bonus za confidence
            score += confidence * 10
            
            # Bonus za tekst liczbowy
            if any(char.isdigit() for char in text_clean):
                score += 100
                valid_elements += 1
                
                # Dodatkowy bonus za notację potęgową lub naukową
                if ('^' in text_clean or 'E+' in text_clean or 'E-' in text_clean or 
                    '10^' in text_clean or '2^' in text_clean):
                    score += 200
                
                # Bonus za długość tekstu liczbowego (więcej cyfr = lepiej)
                digit_count = sum(1 for char in text_clean if char.isdigit())
                score += digit_count * 5
            
            # Kara za bardzo krótki tekst lub pojedyncze znaki
            if len(text_clean) <= 1:
                score -= 50
            
            # Kara za tekst nie-liczbowy (ale nie całkowita dyskwalifikacja)
            elif not any(char.isdigit() for char in text_clean):
                score -= 20
        
        # Bonus za liczbę prawidłowych elementów
        score += valid_elements * 50
        
        # Kara za zbyt mało elementów (prawdopodobnie błędna oś)
        if len(axis) < 3:
            score -= 200
        
        # Bonus za odpowiednią liczbę elementów (typowe dla osi)
        elif 3 <= len(axis) <= 20:
            score += 100
        
        return score
    
    # Znajdź oś z najwyższym wynikiem jakości
    best_axis = max(axes_list, key=axis_quality_score)
    
    # Dodatkowe sprawdzenie - jeśli najlepsza oś ma bardzo niski wynik, zwróć None
    best_score = axis_quality_score(best_axis)
    if best_score < 0:
        return None
    
    return best_axis

def visualize_axes(image, axes_data, output_path):
    """
    Tworzy wizualizację wykrytych osi na obrazie.
    Pokazuje tylko jedną najlepszą oś X i jedną najlepszą oś Y.
    
    Args:
        image: Obraz (np.array) z CV2
        axes_data: Dane o osiach z funkcji detect_axis_groups
        output_path: Ścieżka do zapisania wizualizacji
        
    Returns:
        None, zapisuje obraz do pliku
    """
    visualization = image.copy()
    # Jeśli obraz jest w skali szarości, konwertuj do BGR
    if len(visualization.shape) == 2:
        visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)
    
    # Kolory dla różnych osi (format BGR - Blue, Green, Red) - używamy kontrastowe kolory
    horizontal_color = (0, 255, 0)   # Zielony dla osi X (poziome)
    vertical_color = (255, 0, 0)     # Niebieski dla osi Y (pionowe)
    
    print(f"DEBUG: Rozpoczynam wizualizację osi")
    print(f"DEBUG: Liczba osi poziomych: {len(axes_data['horizontal'])}")
    print(f"DEBUG: Liczba osi pionowych: {len(axes_data['vertical'])}")
    
    # Wybierz najlepsze osie
    best_horizontal = select_best_axis_for_visualization(axes_data['horizontal'])
    best_vertical = select_best_axis_for_visualization(axes_data['vertical'])
    
    # Lista najlepszych osi do rysowania
    best_horizontal_axes = [best_horizontal] if best_horizontal else []
    best_vertical_axes = [best_vertical] if best_vertical else []
    
    print(f"DEBUG: Liczba najlepszych osi poziomych: {len(best_horizontal_axes)}")
    print(f"DEBUG: Liczba najlepszych osi pionowych: {len(best_vertical_axes)}")
    
    # Rysuj najlepszą oś poziomą (X)
    for i, axis in enumerate(best_horizontal_axes):
        print(f"DEBUG: Rysowanie osi poziomej {i+1} z {len(axis)} elementami")
        # Znajdź ogólne granice dla całej osi
        all_x = []
        all_y = []
        
        # Rysuj poszczególne bloki tekstu
        for _, ext_box, orig_box, text, _ in axis:
            # Rysuj oryginalny box z grubszą linią
            points = np.array(orig_box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(visualization, [points], True, horizontal_color, 4)
            
            # Dodaj tekst na środku każdego bloku
            x_coords = [p[0] for p in orig_box]
            y_coords = [p[1] for p in orig_box]
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))
            
            # Rysuj tekst z tłem dla lepszej widoczności
            cv2.putText(visualization, text, (center_x - 15, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Czarne tło
            cv2.putText(visualization, text, (center_x - 15, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, horizontal_color, 1)  # Zielony tekst
            
            # Zbieraj współrzędne do obliczenia granic całej osi
            for p in orig_box:
                all_x.append(p[0])
                all_y.append(p[1])
        
        # Oblicz granice całej osi
        x_min, y_min = min(all_x), min(all_y)
        x_max, y_max = max(all_x), max(all_y)
        
        # Rysuj prostokąt obejmujący całą oś z grubszą linią
        cv2.rectangle(visualization, (int(x_min-5), int(y_min-5)), (int(x_max+5), int(y_max+5)), 
                      horizontal_color, 5, cv2.LINE_AA)
        
        # Dodaj etykietę osi z większym rozmiarem czcionki
        label = f"Oś X{i+1} ({len(axis)} elem.)"
        cv2.putText(visualization, label, (int(x_min), int(y_min - 15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # Czarne tło
        cv2.putText(visualization, label, (int(x_min), int(y_min - 15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, horizontal_color, 2)  # Zielony tekst

    # Rysuj najlepszą oś pionową (Y)
    for i, axis in enumerate(best_vertical_axes):
        print(f"DEBUG: Rysowanie osi pionowej {i+1} z {len(axis)} elementami")
        # Znajdź ogólne granice dla całej osi
        all_x = []
        all_y = []
        
        # Rysuj poszczególne bloki tekstu
        for _, ext_box, orig_box, text, _ in axis:
            # Rysuj oryginalny box z grubszą linią
            points = np.array(orig_box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(visualization, [points], True, vertical_color, 4)
            
            # Dodaj tekst na środku każdego bloku
            x_coords = [p[0] for p in orig_box]
            y_coords = [p[1] for p in orig_box]
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))
            
            # Rysuj tekst z tłem dla lepszej widoczności
            cv2.putText(visualization, text, (center_x + 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)  # Białe tło
            cv2.putText(visualization, text, (center_x + 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, vertical_color, 1)  # Niebieski tekst
            
            # Zbieraj współrzędne do obliczenia granic całej osi
            for p in orig_box:
                all_x.append(p[0])
                all_y.append(p[1])
        
        # Oblicz granice całej osi
        x_min, y_min = min(all_x), min(all_y)
        x_max, y_max = max(all_x), max(all_y)
        
        # Rysuj prostokąt obejmujący całą oś z grubszą linią
        cv2.rectangle(visualization, (int(x_min-5), int(y_min-5)), (int(x_max+5), int(y_max+5)), 
                      vertical_color, 5, cv2.LINE_AA)
        
        # Dodaj etykietę osi z większym rozmiarem czcionki
        label = f"Oś Y{i+1} ({len(axis)} elem.)"
        cv2.putText(visualization, label, (int(x_min - 20), int(y_min)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)  # Białe tło
        cv2.putText(visualization, label, (int(x_min - 20), int(y_min)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, vertical_color, 2)  # Niebieski tekst
    
    # Zapisz wizualizację
    cv2.imwrite(output_path, visualization)
    print(f"DEBUG: Wizualizacja zapisana do: {output_path}")

def format_axes_to_json(axes_data):
    """
    Formatuje dane o osiach do formatu JSON.
    
    Args:
        axes_data: Dane o osiach z funkcji detect_axis_groups
        
    Returns:
        Dict w formacie JSON-friendly
    """
    result = {
        "horizontal_axes": [],
        "vertical_axes": []
    }
    
    # Formatuj osie poziome (X)
    for i, axis in enumerate(axes_data['horizontal']):
        axis_data = {
            "id": f"X{i+1}",
            "elements_count": len(axis),
            "elements": []
        }
        
        # Znajdź ogólne granice dla całej osi
        all_x = []
        all_y = []
        all_texts = []
        
        for _, _, orig_box, text, confidence in axis:
            # Zbieraj współrzędne do obliczenia granic całej osi
            for p in orig_box:
                all_x.append(p[0])
                all_y.append(p[1])
              # Formatuj informacje o pojedynczym elemencie
            x_coords = [p[0] for p in orig_box]
            y_coords = [p[1] for p in orig_box]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            
            # Zastosuj funkcje czyszczenia tekstu
            cleaned_text = clean_duplicated_text([(orig_box, text, confidence)])[0][1]
            cleaned_text = detect_exponent_notation([(orig_box, cleaned_text, confidence)])[0][1]
            
            element = {
                "text": cleaned_text,
                "confidence": float(confidence),
                "bbox": {
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max)
                }
            }
            axis_data["elements"].append(element)
            all_texts.append(cleaned_text)
        
        # Oblicz granice całej osi
        x_min, y_min = min(all_x), min(all_y)
        x_max, y_max = max(all_x), max(all_y)
        
        axis_data["bbox"] = {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max)
        }
        
        axis_data["values"] = all_texts
        result["horizontal_axes"].append(axis_data)
    
    # Formatuj osie pionowe (Y)
    for i, axis in enumerate(axes_data['vertical']):
        axis_data = {
            "id": f"Y{i+1}",
            "elements_count": len(axis),
            "elements": []
        }
        
        # Znajdź ogólne granice dla całej osi
        all_x = []
        all_y = []
        all_texts = []
        
        for _, _, orig_box, text, confidence in axis:
            # Zbieraj współrzędne do obliczenia granic całej osi
            for p in orig_box:
                all_x.append(p[0])
                all_y.append(p[1])
              # Formatuj informacje o pojedynczym elemencie
            x_coords = [p[0] for p in orig_box]
            y_coords = [p[1] for p in orig_box]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            
            # Zastosuj funkcje czyszczenia tekstu
            cleaned_text = clean_duplicated_text([(orig_box, text, confidence)])[0][1]
            cleaned_text = detect_exponent_notation([(orig_box, cleaned_text, confidence)])[0][1]
            
            element = {
                "text": cleaned_text,
                "confidence": float(confidence),
                "bbox": {
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max)
                }
            }
            axis_data["elements"].append(element)
            all_texts.append(cleaned_text)
        
        # Oblicz granice całej osi
        x_min, y_min = min(all_x), min(all_y)
        x_max, y_max = max(all_x), max(all_y)
        
        axis_data["bbox"] = {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max)
        }
        
        axis_data["values"] = all_texts
        result["vertical_axes"].append(axis_data)
    
    return result

def process_image_for_axes(image_path, text_blocks, output_dir, 
                           extension_factor=1.5, min_extension=10,
                           overlap_threshold=0.01, alignment_tolerance=0.3):
    """
    Przetwarza obraz w celu wykrycia osi na podstawie bloków tekstu.
    
    Args:
        image_path: Ścieżka do obrazu
        text_blocks: Lista bloków tekstu w formacie [(bbox, text, confidence), ...]
        output_dir: Katalog do zapisania wyników
        extension_factor: Współczynnik rozszerzenia boksu
        min_extension: Minimalna liczba pikseli do rozszerzenia
        overlap_threshold: Minimalny próg IoU dla uznania nakładania
        alignment_tolerance: Tolerancja wyrównania bloków w osi
        
    Returns:
        Dict zawierający dane o osiach
    """    # Upewnij się, że katalog wyjściowy istnieje
    os.makedirs(output_dir, exist_ok=True)
    
    # Wczytaj obraz
    image = cv2.imread(image_path)
    if image is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return None
      # KROK 1: Rozszerz bounding boxy
    extended_boxes = extend_boxes(text_blocks, extension_factor, min_extension)
    
    # Wykryj grupy osi
    axes_data = detect_axis_groups(extended_boxes, overlap_threshold, alignment_tolerance)
    
    # Przygotuj ścieżki plików wyjściowych
    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)
    
    visualization_path = os.path.join(output_dir, f"{name}_axes.png")
    json_path = os.path.join(output_dir, f"{name}_axes.json")
    
    # Stwórz wizualizację
    visualize_axes(image, axes_data, visualization_path)
    
    # Formatuj dane do JSON i zapisz
    json_data = format_axes_to_json(axes_data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Podsumowanie
    horizontal_count = len(axes_data['horizontal'])
    vertical_count = len(axes_data['vertical'])
    total_elements = sum(len(axis) for axis in axes_data['horizontal'] + axes_data['vertical'])
    
    print(f"Wykryto osie dla {image_path}:")
    print(f"  - Poziome (X): {horizontal_count}")
    print(f"  - Pionowe (Y): {vertical_count}")
    print(f"  - Łącznie elementów: {total_elements}")
    
    return json_data

def process_results_for_axes(json_results_path, output_dir=None, 
                             extension_factor=1.5, min_extension=10,
                             overlap_threshold=0.01, alignment_tolerance=0.3):
    """
    Przetwarza wyniki ekstrakcji tekstu z pliku JSON w celu wykrycia osi.
    
    Args:
        json_results_path: Ścieżka do pliku JSON z wynikami ekstrakcji tekstu
        output_dir: Katalog do zapisania wyników (domyślnie obok pliku JSON w folderze axes)
        extension_factor: Współczynnik rozszerzenia boksu
        min_extension: Minimalna liczba pikseli do rozszerzenia
        overlap_threshold: Minimalny próg IoU dla uznania nakładania
        alignment_tolerance: Tolerancja wyrównania bloków w osi
        
    Returns:
        Dict zawierający dane o osiach
    """
    # Wczytaj dane z pliku JSON
    with open(json_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Pobierz ścieżkę do obrazu i bloków tekstu
    image_path = data["image_path"]
    blocks = data["blocks"]
    
    # Konwertuj dane z JSON do formatu oczekiwanego przez process_image_for_axes
    text_blocks = []
    for block in blocks:
        coords = block["coords"]
        text = block["text"]
        confidence = block["confidence"]
        
        # Konwertuj [x_min, y_min, x_max, y_max] na format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        bbox = [
            [coords[0], coords[1]],  # top-left
            [coords[2], coords[1]],  # top-right
            [coords[2], coords[3]],  # bottom-right
            [coords[0], coords[3]]   # bottom-left
        ]
        
        text_blocks.append((bbox, text, confidence))
    
    # Ustal katalog wyjściowy jeśli nie podano
    if output_dir is None:
        base_dir = os.path.dirname(json_results_path)
        output_dir = os.path.join(base_dir, "axes")
    
    # Przetwórz obraz
    return process_image_for_axes(
        image_path, 
        text_blocks, 
        output_dir,
        extension_factor,
        min_extension,
        overlap_threshold,
        alignment_tolerance
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Wykrywanie osi wykresu na podstawie wykrytego tekstu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("json_file", 
                       help="Ścieżka do pliku JSON z wynikami ekstrakcji tekstu")
    
    parser.add_argument("--output", "-o", 
                       help="Katalog wyjściowy na wyniki (domyślnie: katalog_json/axes)")
    
    parser.add_argument("--extension", "-e", type=float, default=1.5,
                       help="Współczynnik rozszerzenia boksów tekstu")
    
    parser.add_argument("--min-extension", "-m", type=int, default=10,
                       help="Minimalna liczba pikseli do rozszerzenia boksu")
    
    parser.add_argument("--overlap", "-v", type=float, default=0.01,
                       help="Minimalny próg nakładania się boksów (IoU)")
    
    parser.add_argument("--alignment", "-a", type=float, default=0.3,
                       help="Tolerancja wyrównania bloków w osi (0-1)")
    
    args = parser.parse_args()
    
    # Wykonaj detekcję osi
    process_results_for_axes(
        args.json_file, 
        args.output,
        args.extension,
        args.min_extension,
        args.overlap,
        args.alignment
    )
    
    print(f"Wyniki zapisane w katalogu: {args.output or os.path.join(os.path.dirname(args.json_file), 'axes')}") 