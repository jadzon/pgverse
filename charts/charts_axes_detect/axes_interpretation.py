#!/usr/bin/env python3

import os
import json
import re
import numpy as np
from collections import defaultdict

def clean_text_value(text):
    """
    Czyści wartość tekstową, usuwając powtórzenia liczb.
    
    Args:
        text: Tekst do oczyszczenia
        
    Returns:
        Oczyszczony tekst
    """
    # Usuń zbędne znaki i spacje
    text = text.strip()
    
    # Sprawdź, czy tekst zawiera powtórzoną notację naukową (np. "1,2E+09 1,2E+09")
    scientific_pattern = r'(\d+[,.]\d*[Ee][+-]?\d+)\s+\1\b'
    scientific_match = re.search(scientific_pattern, text)
    if scientific_match:
        return scientific_match.group(1)  # Zwróć tylko jedną kopię
    
    # Sprawdź, czy tekst zawiera powtórzone liczby (np. "10 10")
    pattern = r'(\d+)\s+\1\b'
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # Zwróć tylko jedną kopię liczby
    
    # Sprawdź, czy tekst zawiera kilka liczb oddzielonych spacjami
    numbers = re.findall(r'\d+', text)
    if len(numbers) > 1 and len(set(numbers)) == 1:
        return numbers[0]  # Jeśli wszystkie liczby są identyczne, zwróć tylko jedną
    
    return text

def is_likely_axis_label(text, bbox=None):
    """
    Sprawdza czy tekst prawdopodobnie jest etykietą osi a nie wartością.
    Filtruje obrócone litery, jednostki, znaki graficzne itp.
    
    Args:
        text: Tekst do sprawdzenia
        bbox: Słownik z bbox (opcjonalnie) dla sprawdzenia proporcji
        
    Returns:
        bool: True jeśli prawdopodobnie etykieta osi, False jeśli wartość
    """
    if not text:
        return True  # Pusty tekst to prawdopodobnie śmieć
    
    text_clean = text.strip()
    
    # 1. Pojedyncze znaki - prawdopodobnie obrócone litery
    if len(text_clean) == 1:
        # WYJĄTEK: Cyfry 0-9 mogą być wartościami osi, więc NIE filtruj
        if text_clean.isdigit():
            return False  # Pojedyncze cyfry to prawdopodobnie wartości, nie etykiety
        
        # Sprawdź czy to typowe obrócone znaki (ale nie cyfry)
        rotated_chars = {'>', '<', '|', '-', '+', '=', 'y', 'x', 'v', '^'}
        if text_clean in rotated_chars:
            return True
        
        # Specjalny przypadek: "7" może być obróconym "y" - sprawdź kontekst
        if text_clean == '7':
            # Jeśli to wygląda jak obrócone "y", filtruj
            return True
    
    # 2. Operatory porównania i znaki matematyczne
    comparison_operators = {'>', '<', '>=', '<=', '≥', '≤', '=', '!=', '≠'}
    if text_clean in comparison_operators:
        return True
    
    # 3. Teksty z operatorami na początku lub końcu
    # WYJĄTEK: nie filtruj jeśli to operator z liczbą (np. "> 2.0")
    if re.match(r'^[><≥≤=]\s*\d', text_clean) or re.match(r'\d\s*[><≥≤=-]$', text_clean):
        # Sprawdź czy to kompletny operator z liczbą - w takim przypadku nie filtruj
        if re.search(r'[>≥<≤=]\s*\d+(?:[.,]\d+)?$|^\d+(?:[.,]\d+)?\s*[>≥<≤=]', text_clean):
            return False  # To kompletny operator z liczbą - nie filtruj
        return True
    
    # 4. Jednostki miary i etykiety osi (ale NIE liczby)
    unit_patterns = [
        r'\b(ms|sec|min|hr|kg|g|mg|hz|khz|mhz|ghz|%|percent|time|frequency|value|axis)\b',
        r'\b(volt|amp|watt|meter|inch|foot|yard|mile|celsius|fahrenheit|kelvin)\b',
        r'\b(x|y|z|axis|label|title)\b'
    ]
    
    for pattern in unit_patterns:
        if re.search(pattern, text_clean.lower()):
            return True
    
    # 4b. Pojedyncze słowa, ale TYLKO jeśli zawierają litery (nie same cyfry)
    if re.match(r'^[a-zA-Z]+$', text_clean):  # Tylko litery, bez cyfr
        return True
    
    # 5. Sprawdź proporcje bbox - wysokie i wąskie to prawdopodobnie obrócony tekst
    if bbox:
        width = bbox.get('x_max', 0) - bbox.get('x_min', 0)
        height = bbox.get('y_max', 0) - bbox.get('y_min', 0)
        
        if width > 0 and height > 0:
            aspect_ratio = height / width
            # Jeśli wysokość > 3x szerokość, prawdopodobnie obrócony tekst
            if aspect_ratio > 3.0:
                return True
    
    # 6. Tekst zawierający głównie litery - prawdopodobnie etykieta
    # Ale NIE jeśli to mieszanka cyfr i pojedynczych liter (mogą to być obrócone cyfry)
    if re.search(r'[a-zA-Z]', text_clean):
        # Wyjątek dla notacji naukowej (np. "1.2E+09")
        if re.search(r'\d[eE][+-]?\d', text_clean):
            return False  # To notacja naukowa - NIE filtruj
        # Wyjątek dla notacji potęgowej (np. "10^2")
        if re.search(r'\d+\^\d+', text_clean):
            return False  # To notacja potęgowa - NIE filtruj
        # Jeśli więcej liter niż cyfr, prawdopodobnie etykieta
        letters = len(re.findall(r'[a-zA-Z]', text_clean))
        digits = len(re.findall(r'\d', text_clean))
        if letters > digits:
            return True
    
    # 7. Znaki specjalne i symbole (ale nie cudzysłowy przy cyfrach)
    # Wyjątek: cudzysłowy z cyframi mogą być błędami OCR przy rozpoznawaniu liczb
    if re.search(r'^[\'\"]\d+$|^\d+[\'\"]+$', text_clean):
        return False  # To prawdopodobnie błąd OCR - cyfra z cudzysłowem
    
    special_chars = {'$', '€', '£', '¥', '@', '#', '&', '*', '~', '`', '\'', '"'}
    if any(char in text_clean for char in special_chars):
        return True
    
    # 8. Ostatnia sprawdzenie - czy to czysto numeryczne wartości
    # Jeśli tekst zawiera głównie cyfry, przecinki, kropki i znaki +/-, prawdopodobnie to wartość
    if re.match(r'^[\d\.,\-\+\s]+$', text_clean):
        return False  # To liczba - NIE filtruj
    
    # Sprawdź czy tekst zawiera operatory matematyczne z liczbami - te powinny być przetwarzane
    if re.search(r'[>≥<≤=]\s*\d+', text_clean):
        return False
    
    # Jeśli nic nie pasuje, prawdopodobnie to wartość osi
    return False

def clean_ocr_artifacts(text):
    """
    Inteligentnie usuwa artefakty OCR z tekstu liczbowego.
    
    Args:
        text: Tekst do oczyszczenia
        
    Returns:
        str: Oczyszczony tekst
    """
    if not text:
        return text
    
    original_text = text.strip()
    
    # 1. Usuń cudzysłowy z początku i końca cyfr
    text = re.sub(r'^[\'\"]+(\d+(?:[.,]\d+)?)[\'\"]*$', r'\1', text)
    text = re.sub(r'^(\d+(?:[.,]\d+)?)[\'\"]+$', r'\1', text)
    
    # 2. Usuń izolowane litery i znaki w liczbach (prawdopodobne błędy OCR)
    text = re.sub(r'^(\d+(?:[.,]\d+)?)\s*[A-Za-z]+\s*$', r'\1', text)  # "123 ABC" → "123"
    text = re.sub(r'^[A-Za-z]+\s*(\d+(?:[.,]\d+)?)$', r'\1', text)     # "ABC 123" → "123"
    
    # 3. Usuń trailing znaki operatorów i symboli
    text = re.sub(r'^([\d.,+-]+)\s*[-+*/=<>≥≤]+\s*$', r'\1', text)
    
    # 4. Czyszczenie wzorców "cyfra spacja cyfra/litera"
    # Heurystyka: jeśli mamy "X Y" gdzie Y to pojedynczy znak, prawdopodobnie Y to artefakt
    pattern_match = re.match(r'^(\d+(?:[.,]\d+)?)\s+([A-Za-z0-9])$', text)
    if pattern_match:
        main_part = pattern_match.group(1)
        artifact = pattern_match.group(2)
        
        # Specjalne przypadki dla typowych błędów OCR
        if main_part == '7' and artifact == '0':
            # "7  0" → prawdopodobnie obrócone "y" + "0", zostaw samo "0"
            text = artifact
        elif len(main_part) > 1 or (len(main_part) == 1 and artifact.upper() in 'OIL'):
            text = main_part  # Usuń artefakt, zostaw główną część
    
    return text

def extract_numeric_value(text):
    """
    Ekstrahuje wartość liczbową z tekstu.
    
    Args:
        text: Tekst do analizy
        
    Returns:
        float: wartość liczbowa lub None jeśli nie można przekonwertować
    """
    # Najpierw oczyść tekst z powtórzeń
    text = clean_text_value(text)
    
    # Usuń zbędne znaki i spacje
    text = text.strip()
    
    # Inteligentnie oczyść artefakty OCR
    original_text = text
    text = clean_ocr_artifacts(text)
    if text != original_text:
        print(f"    OCR czyszczenie: '{original_text}' → '{text}'")
    
    # Dodatkowa filtracja - sprawdź czy to nie jest etykieta osi
    # ALE tylko dla oczywistych przypadków, nie dla liczb
    if is_likely_axis_label(text) and not re.match(r'^[\d\.,\-\+\s]+$', text):
        return None
    
    # Obsługa tekstów ze znakami porównania (np. "> 2.0", "< 5", "≥ 10")
    # Poprawiony wzorzec, który lepiej wyodrębnia liczby z operatorów
    comparison_pattern = r'[>≥<≤=]+\s*(-?\d+(?:[.,]\d+)?)|(-?\d+(?:[.,]\d+)?)\s*[>≥<≤=]+'
    comparison_match = re.search(comparison_pattern, text)
    
    if comparison_match:
        # Wyciągnij liczbę z którejkolwiek grupy (przed lub po operatorze)
        number_str = comparison_match.group(1) or comparison_match.group(2)
        if number_str:
            number_str = number_str.replace(',', '.')
            try:
                return float(number_str)
            except ValueError:
                pass
    
      # Obsługa notacji potęgowej (np. "10^1", "10^2", "2^8", "2^9" itp.)
    power_notation_pattern = r'(\d+)\^(\d+)'
    power_match = re.search(power_notation_pattern, text)
    
    if power_match:
        base = int(power_match.group(1))
        exponent = int(power_match.group(2))
        return float(base ** exponent)
    
    # Obsługa notacji naukowej (np. "1,2E+09", "1.2E+09" lub "1E+09")
    scientific_notation_pattern = r'(-?\d+(?:[,.]\d*)?)[Ee]([+-]?\d+)'
    scientific_match = re.search(scientific_notation_pattern, text)
    
    if scientific_match:
        base = scientific_match.group(1).replace(',', '.')
        exponent = scientific_match.group(2)
        try:
            return float(base) * (10 ** int(exponent))
        except ValueError:
            pass
    
    # Obsługa liczb z przecinkami/kropkami
    text = text.replace(',', '.')
    
    # Spróbuj wyodrębnić liczbę z tekstu - weź pierwsze wystąpienie liczby
    number_pattern = r'-?\d+\.?\d*'
    number_match = re.search(number_pattern, text)
    
    if number_match:
        try:
            return float(number_match.group(0))
        except ValueError:
            pass
    
    return None

def sort_axis_elements(elements, axis_type='horizontal'):
    """
    Sortuje elementy osi według ich pozycji.
    
    Args:
        elements: Lista elementów osi
        axis_type: 'horizontal' lub 'vertical'
        
    Returns:
        Lista posortowanych elementów
    """
    if axis_type == 'horizontal':
        # Dla osi poziomej sortujemy po x_min (od lewej do prawej)
        return sorted(elements, key=lambda e: e['bbox']['x_min'])
    else:
        # Dla osi pionowej sortujemy po y_min (od góry do dołu)
        return sorted(elements, key=lambda e: e['bbox']['y_min'])

def convert_numpy_types(obj):
    """
    Konwertuje typy NumPy na standardowe typy Pythona dla serializacji JSON.
    
    Args:
        obj: Obiekt do konwersji
        
    Returns:
        Obiekt skonwertowany do typów standardowych
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

def contextual_text_cleanup(text_values):
    """
    Analizuje wszystkie wartości tekstowe osi razem i czyści błędy OCR w kontekście.
    
    Args:
        text_values: Lista tekstów z osi
        
    Returns:
        Lista oczyszczonych tekstów
    """
    if not text_values:
        return text_values
    
    cleaned_values = []
    
    for text in text_values:
        cleaned = clean_ocr_artifacts(text)
        
        # Dodatkowa analiza kontekstowa
        # Jeśli mamy wiele podobnych wzorców błędów, rozpoznaj je
        
        # Wzorzec 1: "cyfra spacja 0" może być błędnie rozpoznanym "0"
        if re.match(r'^\d\s+0$', cleaned) and text != cleaned:
            # Sprawdź czy w kontekście są inne pojedyncze cyfry
            other_singles = [t for t in text_values if re.match(r'^\d$', clean_ocr_artifacts(t))]
            if len(other_singles) > 1:  # Jeśli są inne pojedyncze cyfry, prawdopodobnie to też powinna być jedna
                cleaned = re.sub(r'^\d\s+(\d)$', r'\1', text)
        
        cleaned_values.append(cleaned)
    
    return cleaned_values

def detect_axis_step(elements, axis_type='horizontal'):
    """
    Wykrywa kroki (interwały) między wartościami na osi.
    
    Args:
        elements: Lista elementów osi
        axis_type: 'horizontal' lub 'vertical'
        
    Returns:
        Dict z informacjami o krokach i wartościach
    """
    # Sortuj elementy według pozycji
    sorted_elements = sort_axis_elements(elements, axis_type)
    
    # Najpierw zbierz wszystkie teksty do analizy kontekstowej
    all_texts = [elem['text'] for elem in sorted_elements if not is_likely_axis_label(elem['text'], elem['bbox'])]
    
    # Przeprowadź kontekstowe czyszczenie
    cleaned_texts = contextual_text_cleanup(all_texts)
    
    # Wyodrębnij wartości liczbowe
    values = []
    positions = []
    text_values = []
    
    text_index = 0
    for elem in sorted_elements:
        # Sprawdź czy to prawdopodobnie etykieta osi (a nie wartość)
        if is_likely_axis_label(elem['text'], elem['bbox']):
            print(f"  Pomijam prawdopodobną etykietę osi: '{elem['text']}'")
            continue
        

        
        # Użyj oczyszczonego tekstu
        cleaned_text = cleaned_texts[text_index] if text_index < len(cleaned_texts) else elem['text']
        text_index += 1
        
        value = extract_numeric_value(cleaned_text)
        if value is not None:
            values.append(value)
            text_values.append(cleaned_text)
            
            if axis_type == 'horizontal':
                # Dla osi poziomej bierzemy środek elementu w osi X
                x_center = (elem['bbox']['x_min'] + elem['bbox']['x_max']) / 2
                positions.append(x_center)
            else:
                # Dla osi pionowej bierzemy środek elementu w osi Y
                y_center = (elem['bbox']['y_min'] + elem['bbox']['y_max']) / 2
                positions.append(y_center)
    
    # Jeśli mamy za mało wartości, nie możemy określić kroku
    if len(values) < 2:
        return {
            'status': 'error',
            'message': 'Za mało wartości liczbowych na osi',
            'values': text_values
        }
    
    # Oblicz różnice między kolejnymi wartościami sprawdzając

    value_differences = [values[i+1] - values[i] for i in range(len(values)-1)]
    position_differences = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    # Oblicz średni krok wartości i pozycji (konwertuj z NumPy do Python)
    avg_value_step = float(np.mean(value_differences))
    avg_position_step = float(np.mean(position_differences))
    
    # Oblicz odchylenie standardowe kroków wartości i pozycji
    value_step_std = float(np.std(value_differences))
    position_step_std = float(np.std(position_differences))
    
    # Sprawdź, czy kroki są w miarę równomierne
    # Oblicz medianę różnic dla większej odporności na wartości odstające
    median_value_step = float(np.median(value_differences))
    median_position_step = float(np.median(position_differences))
    print(f"Median value step: {median_value_step}, Median position step: {median_position_step}")
    # Sprawdź anomalie które mogą być błędami OCR
    if len(value_differences) >= 3:
        # Znajdź różnice dramatycznie odbiegające od mediany
        potential_errors = [i for i, diff in enumerate(value_differences) 
                           if abs(diff - median_value_step) > 2 * abs(median_value_step)]
        
        if potential_errors and len(potential_errors) <= len(value_differences) // 3:
            print(f"Wykryto potencjalne błędy OCR na indeksach: {potential_errors}")
            # Oblicz odchylenie standardowe bez wartości odstających
            clean_diffs = [diff for i, diff in enumerate(value_differences) 
                          if i not in potential_errors]
            if clean_diffs:
                value_step_std = float(np.std(clean_diffs))
                avg_value_step = float(np.mean(clean_diffs))
                print(f"Skorygowano: Średni krok: {avg_value_step}, Odchylenie: {value_step_std}")

    # Bardziej odporna kontrola równomierności
    is_uniform = ((value_step_std < 0.01) or 
                  (value_step_std / abs(avg_value_step) < 0.3)) if avg_value_step != 0 else False
    is_pixel_uniform = ((position_step_std < 0.01) or 
                       (position_step_std / abs(avg_position_step) < 0.3)) if avg_position_step != 0 else False
    # Sprawdź, czy oś jest logarytmiczna
    is_logarithmic = False
    logarithm_base = None # Domyślna podstawa logarytmu
    

    if len(values) >= 3 and all(v > 0 for v in values):  # Logarytm działa tylko dla wartości dodatnich
        if not is_uniform:
            # Prawdopodobnie oś logarytmiczna
            is_logarithmic = True
            
            # Oblicz podstawę logarytmu
            # Dla typowych podstaw (10, 2, e) sprawdź która najlepiej pasuje
            log_bases = [2.0, np.e, 10.0]
            best_base = 10.0
            min_error = float('inf')
            
            for base in log_bases:
                # Przekształć wartości na logarytmy o danej podstawie
                log_values = [np.log(v) / np.log(base) for v in values]
                
                # Sprawdź liniowość między pozycjami a logarytmami wartości
                # Oblicz regresję liniową
                slope, intercept = np.polyfit(positions, log_values, 1)
                
                # Oblicz błąd predykcji
                predicted = [slope * p + intercept for p in positions]
                error = np.sum([(predicted[i] - log_values[i])**2 for i in range(len(log_values))])
                print(f"Base {base}: Error = {error}")
                if error < min_error:
                    min_error = error
                    best_base = base
            
            logarithm_base = float(best_base)
            
            # Dla osi logarytmicznej, piksele na jednostkę wartości to piksele na jednostkę logarytmu
            log_values = [np.log(v) / np.log(logarithm_base) for v in values]
            log_diffs = [log_values[i+1] - log_values[i] for i in range(len(log_values)-1)]
            avg_log_step = float(np.mean(log_diffs))
            pixels_per_log_unit = float(avg_position_step / avg_log_step) if avg_log_step != 0 else 0
    # Dla osi pionowej, kierunek wartości jest zwykle odwrócony (większe wartości na górze)
    # czyli większa wartość = mniejsza współrzędna Y
    # Jeśli wartości maleją gdy pozycje rosną (iloczyn negatywny), to oś jest odwrócona
    if axis_type == 'vertical' and avg_value_step * avg_position_step < 0:
        direction = 'reversed'  # Wartości rosną w dół (odwrócone)
    else:
        direction = 'normal'    # Wartości rosną standardowo
    
    # Oblicz krok pikseli na jednostkę wartości (współczynnik skalowania)
    # Dla bardzo dużych wartości (np. 1E+09) używamy skali logarytmicznej
    max_value = max(values)
    
    scale_factor = 1.0

    # Jeśli wartości są duże, normalizujemy do zakresu 0-100
    if max_value > 1000 and not is_logarithmic:
        scale_factor = 100.0 / max_value
        normalized_values = [v * scale_factor for v in values]
        print(f"Normalized values: {normalized_values}")
        normalized_differences = [normalized_values[i+1] - normalized_values[i] for i in range(len(normalized_values)-1)]

        normalized_step = np.mean(normalized_differences)
        pixels_per_unit = abs(avg_position_step / normalized_step) if normalized_step != 0 else 0
    elif not is_logarithmic:
        pixels_per_unit = abs(avg_position_step / avg_value_step) if avg_value_step != 0 else 0
    else:
        # Dla osi logarytmicznej, już obliczono pixels_per_log_unit
        pixels_per_unit = abs(pixels_per_log_unit)

    # Usuń duplikaty - te same wartości na zbliżonych pozycjach (błędy OCR)
    def remove_duplicates(vals, texts, poss, pixel_threshold=50):
        """Usuwa duplikaty wartości znajdujące się na zbliżonych pozycjach
        
        Args:
            vals: Lista wartości
            texts: Lista tekstów
            poss: Lista pozycji
            pixel_threshold: Próg odległości w pikselach (domyślnie 50)
        """
        if len(vals) <= 1:
            return vals, texts, poss
            
        clean_vals = []
        clean_texts = []
        clean_poss = []
        
        for i, (val, text, pos) in enumerate(zip(vals, texts, poss)):
            # Sprawdź czy ta wartość już istnieje na zbliżonej pozycji
            is_duplicate = False
            for j, (existing_val, existing_text, existing_pos) in enumerate(zip(clean_vals, clean_texts, clean_poss)):
                if (val == existing_val and 
                    abs(pos - existing_pos) < pixel_threshold):
                    print(f"  🗑️ Usuwam duplikat: {text} na pozycji {pos} (już mamy {existing_text} na {existing_pos})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                clean_vals.append(val)
                clean_texts.append(text)
                clean_poss.append(pos)
        
        return clean_vals, clean_texts, clean_poss
    
    # Zastosuj usuwanie duplikatów przed automatycznym odwracaniem
    values, text_values, positions = remove_duplicates(values, text_values, positions)
    
    # Automatycznie odwróć wartości osi Y jeśli kierunek jest odwrócony
    final_values = values.copy()
    final_text_values = text_values.copy()
    final_positions = positions.copy()
    
    if direction == 'reversed' and axis_type == 'vertical':
        # Odwróć kolejność wartości dla osi Y aby były zgodne z matematyczną konwencją (rosnąco od dołu)
        final_values = list(reversed(values))
        final_text_values = list(reversed(text_values))
        final_positions = list(reversed(positions))
        print(f"🔄 Automatycznie odwrócono kolejność osi Y: {text_values} → {final_text_values}")
        direction = 'normal'  # Po odwróceniu kierunek staje się normalny

    # Przygotuj wynik
    result = {
        'status': 'success',
        'values': [float(v) for v in final_values],  # Konwertuj wszystkie wartości na float Pythona
        'text_values': final_text_values,
        'positions': [float(p) for p in final_positions],  # Konwertuj wszystkie pozycje na float Pythona
        'step': float(abs(avg_value_step)),              # Krok wartości (np. 1.0, 5.0)
        'pixel_step': float(abs(avg_position_step)),     # Krok w pikselach między kolejnymi wartościami
        'pixels_per_unit': float(pixels_per_unit),       # Piksele na jednostkę wartości
        'scale_factor': float(scale_factor),             # Współczynnik skalowania (1.0 dla normalnych wartości)
        'is_uniform': is_uniform,                        # Czy kroki wartości są równomierne
        'is_pixel_uniform': is_pixel_uniform,            # Czy kroki pikseli są równomierne
        'direction': direction,
        'is_logarithmic': is_logarithmic,                # Czy oś jest logarytmiczna  
        'logarithm_base': logarithm_base,                # Podstawa logarytmu, jeśli oś jest logarytmiczna
    }
    
    return result

def select_best_axis(axes):
    """
    Wybiera najlepszą oś spośród dostępnych na podstawie kryteriów jakości.
    
    Args:
        axes: Lista interpretacji osi
        
    Returns:
        Najlepsza interpretacja osi lub None jeśli żadna nie jest dostępna
    """
    if not axes:
        return None
    
    # Filtruj tylko pomyślne interpretacje
    successful_axes = [axis for axis in axes if axis.get('status') == 'success']
    
    if not successful_axes:
        # Jeśli żadna nie jest pomyślna, zwróć pierwszą (może zawierać informacje o błędzie)
        return axes[0]
    
    # Kryteria wyboru najlepszej osi (w kolejności ważności):
    # 1. Najwięcej wartości tekstowych
    # 2. Najmniejszy krok (bardziej szczegółowa)
    # 3. Jednolite kroki
    
    def axis_score(axis):
        score = 0
        
        # Liczba wartości (najwięcej punktów)
        if 'text_values' in axis:
            score += len(axis['text_values']) * 100
        
        # Jednolite kroki (bonus)
        if axis.get('is_uniform', False):
            score += 50
        
        # Mniejszy krok = bardziej szczegółowa oś (odwracamy, żeby mniejszy krok = wyższy wynik)
        if 'step' in axis and axis['step'] > 0:
            score += 1.0 / (axis['step'] + 0.001)
        
        return score
    
    # Znajdź oś z najwyższym wynikiem
    best_axis = max(successful_axes, key=axis_score)
    return best_axis

def interpret_axes(axes_data):
    """
    Interpretuje dane osi, wykrywając kroki i zakres wartości.
    Wybiera tylko jedną najlepszą oś X i jedną najlepszą oś Y.
    
    Args:
        axes_data: Dane osi w formacie JSON z funkcji format_axes_to_json
        
    Returns:
        Dict z interpretacją osi (maksymalnie jedna oś X i jedna oś Y)
    """
    horizontal_interpretations = []
    vertical_interpretations = []
    
    # Przetwórz wszystkie osie poziome (X)
    for axis in axes_data.get('horizontal_axes', []):
        axis_id = axis['id']
        elements = axis['elements']
        
        interpretation = detect_axis_step(elements, 'horizontal')
        interpretation['axis_id'] = axis_id
        
        # Dodaj zakres wartości
        if 'values' in interpretation and len(interpretation['values']) > 0:
            try:
                # Filtruj wartości - tylko te które są liczbami
                numeric_values = [v for v in interpretation['values'] if isinstance(v, (int, float))]
                if numeric_values:
                    interpretation['range'] = {
                        'min': float(min(numeric_values)),
                        'max': float(max(numeric_values))
                    }
                else:
                    interpretation['range'] = {'min': 0.0, 'max': 1.0}
            except (ValueError, TypeError) as e:
                print(f"Błąd przy obliczaniu zakresu dla osi poziomej {axis_id}: {e}")
                interpretation['range'] = {'min': 0.0, 'max': 1.0}
            
        # Dodaj oczyszczone wartości tekstowe
        if 'text_values' in interpretation:
            interpretation['cleaned_values'] = [clean_text_value(val) for val in interpretation['text_values']]
        
        horizontal_interpretations.append(interpretation)
    
    # Przetwórz wszystkie osie pionowe (Y)
    for axis in axes_data.get('vertical_axes', []):
        axis_id = axis['id']
        elements = axis['elements']
        
        interpretation = detect_axis_step(elements, 'vertical')
        interpretation['axis_id'] = axis_id
        
        # Dodaj zakres wartości
        if 'values' in interpretation and len(interpretation['values']) > 0:
            try:
                # Filtruj wartości - tylko te które są liczbami
                numeric_values = [v for v in interpretation['values'] if isinstance(v, (int, float))]
                if numeric_values:
                    interpretation['range'] = {
                        'min': float(min(numeric_values)),
                        'max': float(max(numeric_values))
                    }
                else:
                    interpretation['range'] = {'min': 0.0, 'max': 1.0}
            except (ValueError, TypeError) as e:
                print(f"Błąd przy obliczaniu zakresu dla osi pionowej {axis_id}: {e}")
                interpretation['range'] = {'min': 0.0, 'max': 1.0}
            
        # Dodaj oczyszczone wartości tekstowe
        if 'text_values' in interpretation:
            interpretation['cleaned_values'] = [clean_text_value(val) for val in interpretation['text_values']]
        
        vertical_interpretations.append(interpretation)
    
    # Wybierz najlepsze osie
    result = {
        'horizontal_axes': [],
        'vertical_axes': []
    }
    
    # Wybierz najlepszą oś poziomą
    best_horizontal = select_best_axis(horizontal_interpretations)
    if best_horizontal:
        result['horizontal_axes'].append(best_horizontal)
    
    # Wybierz najlepszą oś pionową
    best_vertical = select_best_axis(vertical_interpretations)
    if best_vertical:
        result['vertical_axes'].append(best_vertical)
    
    # Konwertuj wszystkie typy NumPy na standardowe typy Pythona
    return convert_numpy_types(result)

def process_axes_interpretation(axes_json_path, output_dir=None):
    """
    Przetwarza plik JSON z danymi osi i interpretuje wartości.
    
    Args:
        axes_json_path: Ścieżka do pliku JSON z danymi osi
        output_dir: Katalog wyjściowy (domyślnie ten sam co plik wejściowy)
        
    Returns:
        Dict z interpretacją osi
    """
    # Wczytaj dane osi
    with open(axes_json_path, 'r', encoding='utf-8') as f:
        axes_data = json.load(f)
    
    # Interpretuj osie
    interpretation = interpret_axes(axes_data)
    
    # Zapisz wyniki
    if output_dir is None:
        output_dir = os.path.dirname(axes_json_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(axes_json_path)
    name, _ = os.path.splitext(base_name)
    
    # Usuń "_axes" z nazwy pliku, jeśli istnieje
    if name.endswith('_axes'):
        name = name[:-5]
    
    output_path = os.path.join(output_dir, f"{name}_axes_interpretation.json")
    
    # Zapisz dane do JSON, upewniając się, że wszystkie wartości są serializowalne
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(interpretation, f, indent=2, ensure_ascii=False)
    
    # Wypisz podsumowanie
    horizontal_count = len(interpretation['horizontal_axes'])
    vertical_count = len(interpretation['vertical_axes'])
    
    print(f"Interpretacja osi dla {axes_json_path}:")
    print(f"  - Zinterpretowano {horizontal_count} osi poziomych (X)")
    print(f"  - Zinterpretowano {vertical_count} osi pionowych (Y)")
    
    # Wypisz szczegóły dla osi poziomych
    for i, axis in enumerate(interpretation['horizontal_axes']):
        if axis['status'] == 'success':
            # Oczyszczone wartości tekstowe
            clean_values = [clean_text_value(val) for val in axis['text_values']]
            print(f"  - Oś {axis['axis_id']}:")
            print(f"    - Oryginalne wartości: {axis['text_values']}")
            print(f"    - Oczyszczone wartości: {clean_values}")
            print(f"    - Krok wartości: {axis['step']:.2f}")
            print(f"    - Krok w pikselach: {axis['pixel_step']:.2f} px")
            print(f"    - Piksele na jednostkę: {axis['pixels_per_unit']:.2f} px/jednostkę")
            print(f"    - Zakres: {axis['range']['min']:.2f} - {axis['range']['max']:.2f}")
            print(f"    - Jednolite kroki wartości: {'Tak' if axis['is_uniform'] else 'Nie'}")
            print(f"    - Jednolite kroki pikseli: {'Tak' if axis['is_pixel_uniform'] else 'Nie'}")
        else:
            print(f"  - Oś {axis['axis_id']}: {axis['message']}")
    
    # Wypisz szczegóły dla osi pionowych
    for i, axis in enumerate(interpretation['vertical_axes']):
        if axis['status'] == 'success':
            # Oczyszczone wartości tekstowe
            clean_values = [clean_text_value(val) for val in axis['text_values']]
            print(f"  - Oś {axis['axis_id']}:")
            print(f"    - Oryginalne wartości: {axis['text_values']}")
            print(f"    - Oczyszczone wartości: {clean_values}")
            print(f"    - Krok wartości: {axis['step']:.2f}")
            print(f"    - Krok w pikselach: {axis['pixel_step']:.2f} px")
            print(f"    - Piksele na jednostkę: {axis['pixels_per_unit']:.2f} px/jednostkę")
            print(f"    - Zakres: {axis['range']['min']:.2f} - {axis['range']['max']:.2f}")
            print(f"    - Jednolite kroki wartości: {'Tak' if axis['is_uniform'] else 'Nie'}")
            print(f"    - Jednolite kroki pikseli: {'Tak' if axis['is_pixel_uniform'] else 'Nie'}")
            print(f"    - Kierunek: {'Odwrócony (wartości rosną w dół)' if axis['direction'] == 'reversed' else 'Normalny'}")
        else:
            print(f"  - Oś {axis['axis_id']}: {axis['message']}")
    
    print(f"Zapisano interpretację do: {output_path}")
    
    return interpretation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interpretacja wartości na osiach wykresów",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("axes_json", 
                       help="Ścieżka do pliku JSON z danymi osi")
    
    parser.add_argument("--output", "-o", 
                       help="Katalog wyjściowy na wyniki (domyślnie: ten sam co plik wejściowy)")
    
    args = parser.parse_args()
    
    # Wykonaj interpretację osi
    process_axes_interpretation(args.axes_json, args.output)