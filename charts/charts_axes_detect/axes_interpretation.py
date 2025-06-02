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

def extract_numeric_value(text):
    """
    Ekstrahuje wartość liczbową z tekstu.
    
    Args:
        text: Tekst do analizy
        
    Returns:
        float: wartość liczbowa lub None jeśli nie można przekonwertować
    """    # Najpierw oczyść tekst z powtórzeń
    text = clean_text_value(text)
    
    # Usuń zbędne znaki i spacje
    text = text.strip()
      # Obsługa notacji potęgowej (np. "10^1", "10^2", "2^8", "2^9" itp.)
    power_notation_pattern = r'(\d+)\^(\d+)'
    power_match = re.search(power_notation_pattern, text)
    
    if power_match:
        base = int(power_match.group(1))
        exponent = int(power_match.group(2))
        return float(base ** exponent)
    
    # Obsługa notacji naukowej (np. "1,2E+09" lub "1.2E+09")
    scientific_notation_pattern = r'(-?\d+[,.]\d*)[Ee]([+-]?\d+)'
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
    
    # Wyodrębnij wartości liczbowe
    values = []
    positions = []
    text_values = []
    
    for elem in sorted_elements:
        value = extract_numeric_value(elem['text'])
        if value is not None:
            values.append(value)
            text_values.append(elem['text'])
            
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
    is_uniform = bool(value_step_std / abs(avg_value_step) < 0.3) if avg_value_step != 0 else False
    is_pixel_uniform = bool(position_step_std / abs(avg_position_step) < 0.3) if avg_position_step != 0 else False
    # Sprawdź, czy oś jest logarytmiczna
    is_logarithmic = False
    logarithm_base = 10.0  # Domyślna podstawa logarytmu
    
    if len(values) >= 3 and all(v > 0 for v in values):  # Logarytm działa tylko dla wartości dodatnich
        # Sprawdź czy stosunek sąsiednich wartości jest względnie stały
        value_ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
        
        # Oblicz odchylenie standardowe stosunków
        value_ratio_std = float(np.std(value_ratios))
        value_ratio_mean = float(np.mean(value_ratios))
        
        # Sprawdź czy stosunki są w miarę równomierne, a kroki pikseli też
        if (value_ratio_std / value_ratio_mean < 0.5) and is_pixel_uniform:
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
    if axis_type == 'vertical' and avg_value_step * avg_position_step > 0:
        direction = 'reversed'  # Wartości rosną w dół
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
        normalized_differences = [normalized_values[i+1] - normalized_values[i] for i in range(len(normalized_values)-1)]
        normalized_step = np.mean(normalized_differences)
        pixels_per_unit = abs(avg_position_step / normalized_step) if normalized_step != 0 else 0
    elif not is_logarithmic:
        pixels_per_unit = abs(avg_position_step / avg_value_step) if avg_value_step != 0 else 0
    else:
        # Dla osi logarytmicznej, już obliczono pixels_per_log_unit
        pixels_per_unit = pixels_per_log_unit
    
    # Przygotuj wynik
    result = {
        'status': 'success',
        'values': [float(v) for v in values],  # Konwertuj wszystkie wartości na float Pythona
        'text_values': text_values,
        'positions': [float(p) for p in positions],  # Konwertuj wszystkie pozycje na float Pythona
        'step': float(abs(avg_value_step)),              # Krok wartości (np. 1.0, 5.0)
        'pixel_step': float(abs(avg_position_step)),     # Krok w pikselach między kolejnymi wartościami
        'pixels_per_unit': float(pixels_per_unit),       # Piksele na jednostkę wartości
        'scale_factor': float(scale_factor),             # Współczynnik skalowania (1.0 dla normalnych wartości)
        'is_uniform': is_uniform,                        # Czy kroki wartości są równomierne
        'is_pixel_uniform': is_pixel_uniform,            # Czy kroki pikseli są równomierne
        'direction': direction,
        'is_logarithmic': is_logarithmic
    }
    
    return result

def interpret_axes(axes_data):
    """
    Interpretuje dane osi, wykrywając kroki i zakres wartości.
    
    Args:
        axes_data: Dane osi w formacie JSON z funkcji format_axes_to_json
        
    Returns:
        Dict z interpretacją osi
    """
    result = {
        'horizontal_axes': [],
        'vertical_axes': []
    }
    
    # Przetwórz osie poziome (X)
    for axis in axes_data.get('horizontal_axes', []):
        axis_id = axis['id']
        elements = axis['elements']
        
        interpretation = detect_axis_step(elements, 'horizontal')
        interpretation['axis_id'] = axis_id
        
        # Dodaj zakres wartości
        if 'values' in interpretation and len(interpretation['values']) > 0:
            interpretation['range'] = {
                'min': float(min(interpretation['values'])),
                'max': float(max(interpretation['values']))
            }
            
        # Dodaj oczyszczone wartości tekstowe
        if 'text_values' in interpretation:
            interpretation['cleaned_values'] = [clean_text_value(val) for val in interpretation['text_values']]
        
        result['horizontal_axes'].append(interpretation)
    
    # Przetwórz osie pionowe (Y)
    for axis in axes_data.get('vertical_axes', []):
        axis_id = axis['id']
        elements = axis['elements']
        
        interpretation = detect_axis_step(elements, 'vertical')
        interpretation['axis_id'] = axis_id
        
        # Dodaj zakres wartości
        if 'values' in interpretation and len(interpretation['values']) > 0:
            interpretation['range'] = {
                'min': float(min(interpretation['values'])),
                'max': float(max(interpretation['values']))
            }
            
        # Dodaj oczyszczone wartości tekstowe
        if 'text_values' in interpretation:
            interpretation['cleaned_values'] = [clean_text_value(val) for val in interpretation['text_values']]
        
        result['vertical_axes'].append(interpretation)
    
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