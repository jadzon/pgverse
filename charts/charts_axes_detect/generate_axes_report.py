#!/usr/bin/env python3

import os
import json
import glob
from pathlib import Path
import numpy as np
import re

# Stałe konfiguracyjne - łatwo dostosowywalne progi
NUMERICAL_TOLERANCE = 0.01  # Tolerancja dla porównań numerycznych (1%)
HIGH_ACCURACY_THRESHOLD = 0.9  # Próg wysokiej dokładności pozycyjnej (90%)
POSITIONAL_WEIGHT = 0.7  # Waga dokładności pozycyjnej w końcowym wyniku (70%)
CONTENT_WEIGHT = 0.3    # Waga dokładności zawartości w końcowym wyniku (30%)
DIRECTION_CONFIDENCE_THRESHOLD = 0.8  # Próg pewności wykrywania kierunku (80%)

def load_ground_truth(ground_truth_file="charts_axes_detect/true_axes.txt"):
    """
    Wczytuje prawdziwe wartości osi z pliku tekstowego.
    
    Args:
        ground_truth_file: Ścieżka do pliku z prawdziwymi wartościami
        
    Returns:
        Dict z prawdziwymi wartościami dla każdego obrazu
    """
    ground_truth = {}
    
    if not os.path.exists(ground_truth_file):
        print(f"⚠️  Plik z prawdziwymi wartościami nie istnieje: {ground_truth_file}")
        return ground_truth
    
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        i = 0
        while i < len(lines):
            # Linia z nazwą obrazu (numerowaną lub test*)
            image_name = lines[i]
            ground_truth[image_name] = {'X': [], 'Y': []}
            i += 1
            
            # Wczytuj linie z osiami X i Y
            while i < len(lines) and not lines[i].isdigit() and not lines[i].startswith('test'):
                line = lines[i]
                if line.startswith('X '):
                    # Parsuj wartości osi X
                    x_values_str = line[2:].strip()  # Usuń "X "
                    x_values = []
                    for val in x_values_str.split():
                        try:
                            # Spróbuj najpierw int, potem float
                            if '.' in val or 'E' in val.upper() or '^' in val:
                                if '^' in val:
                                    # Obsłuż notację potęgową jak 10^2
                                    base, exp = val.split('^')
                                    x_values.append(float(int(base) ** int(exp)))
                                elif 'E' in val.upper():
                                    # Obsłuż notację naukową
                                    x_values.append(float(val.replace(',', '.')))
                                else:
                                    x_values.append(float(val))
                            else:
                                x_values.append(int(val))
                        except ValueError:
                            # Jeśli nie można sparsować, traktuj jako string
                            x_values.append(val)
                    ground_truth[image_name]['X'] = x_values
                    
                elif line.startswith('Y '):
                    # Parsuj wartości osi Y
                    y_values_str = line[2:].strip()  # Usuń "Y "
                    y_values = []
                    for val in y_values_str.split():
                        try:
                            # Spróbuj najpierw int, potem float
                            if '.' in val or 'E' in val.upper() or '^' in val:
                                if '^' in val:
                                    # Obsłuż notację potęgową jak 2^5
                                    base, exp = val.split('^')
                                    y_values.append(float(int(base) ** int(exp)))
                                elif 'E' in val.upper():
                                    # Obsłuż notację naukową
                                    y_values.append(float(val.replace(',', '.')))
                                else:
                                    y_values.append(float(val))
                            else:
                                y_values.append(int(val))
                        except ValueError:
                            # Jeśli nie można sparsować, traktuj jako string
                            y_values.append(val)
                    ground_truth[image_name]['Y'] = y_values
                i += 1
            
            # Jeśli następna linia to nazwa obrazu, nie zwiększaj i
            if i < len(lines) and (lines[i].isdigit() or lines[i].startswith('test')):
                continue
                
    except Exception as e:
        print(f"❌ Błąd przy wczytywaniu pliku {ground_truth_file}: {e}")
    
    return ground_truth

def compare_axes_values(detected_values, true_values, tolerance=NUMERICAL_TOLERANCE):
    """
    Porównuje wykryte wartości z prawdziwymi UWZGLĘDNIAJĄC KOLEJNOŚĆ I KIERUNEK OSI.
    Automatycznie wykrywa czy oś jest rosnąca czy malejąca i dopasowuje kolejność.
    
    Args:
        detected_values: Lista wykrytych wartości (w kolejności wykrycia)
        true_values: Lista prawdziwych wartości (w oczekiwanej kolejności)
        tolerance: Tolerancja dla porównania wartości zmiennoprzecinkowych
        
    Returns:
        Dict z wynikami porównania
    """
    if not detected_values or not true_values:
        return {
            'accuracy': 0.0,
            'detected_count': len(detected_values) if detected_values else 0,
            'true_count': len(true_values) if true_values else 0,
            'matches': [],
            'missing': true_values if true_values else [],
            'extra': detected_values if detected_values else [],
            'order_correct': False,
            'direction_match': False,
            'auto_reversed': False
        }
    
    # Konwertuj do float jeśli możliwe
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val
    
    detected_float = [safe_float(v) for v in detected_values]
    true_float = [safe_float(v) for v in true_values]
    
    # Wykryj kierunek (rosnący/malejący) dla wartości numerycznych
    def detect_direction(values):
        """Wykrywa czy lista wartości jest rosnąca (1), malejąca (-1), czy niejednoznaczna (0)"""
        if len(values) < 2:
            return 0
        
        numeric_values = []
        for v in values:
            if isinstance(v, (int, float)):
                numeric_values.append(v)
        
        if len(numeric_values) < 2:
            return 0
        
        # Sprawdź trend - policz ile par jest rosnących vs malejących
        rising_pairs = 0
        falling_pairs = 0
        
        for i in range(len(numeric_values) - 1):
            if numeric_values[i] < numeric_values[i + 1]:
                rising_pairs += 1
            elif numeric_values[i] > numeric_values[i + 1]:
                falling_pairs += 1
        
        # Jeśli > 80% par ma ten sam trend, uznaj za jednoznaczny kierunek
        total_pairs = rising_pairs + falling_pairs
        if total_pairs == 0:
            return 0
        
        if rising_pairs / total_pairs >= DIRECTION_CONFIDENCE_THRESHOLD:
            return 1  # Rosnąca
        elif falling_pairs / total_pairs >= DIRECTION_CONFIDENCE_THRESHOLD:
            return -1  # Malejąca
        else:
            return 0  # Niejednoznaczna
    
    true_direction = detect_direction(true_float)
    detected_direction = detect_direction(detected_float)
    
    # Sprawdź czy potrzeba odwrócić wykryte wartości
    auto_reversed = False
    working_detected = detected_float.copy()
    
    if true_direction != 0 and detected_direction != 0 and true_direction != detected_direction:
        # Kierunki są przeciwne - odwróć wykryte wartości
        working_detected = list(reversed(detected_float))
        auto_reversed = True
    
    # Sprawdź dokładność pozycyjną (element za elementem)
    def compare_position_by_position(det_vals, true_vals):
        if len(det_vals) != len(true_vals):
            return 0, []
        
        matches = []
        correct_positions = 0
        
        for i, (det_val, true_val) in enumerate(zip(det_vals, true_vals)):
            is_match = False
            if isinstance(true_val, (int, float)) and isinstance(det_val, (int, float)):
                # Porównanie numeryczne z tolerancją
                if abs(true_val - det_val) <= tolerance * max(abs(true_val), abs(det_val), 1):
                    is_match = True
            else:
                # Porównanie tekstowe
                if str(true_val) == str(det_val):
                    is_match = True
            
            if is_match:
                matches.append((true_val, det_val))
                correct_positions += 1
        
        return correct_positions, matches
    
    # Sprawdź dokładność pozycyjną
    correct_positions, position_matches = compare_position_by_position(working_detected, true_float)
    position_accuracy = correct_positions / len(true_float) if true_float else 0.0
    
    # Jeśli pozycyjna dokładność jest wysoka (>= 90%), uznaj za sukces
    if position_accuracy >= HIGH_ACCURACY_THRESHOLD:
        return {
            'accuracy': position_accuracy,
            'detected_count': len(detected_values),
            'true_count': len(true_values),
            'matches': position_matches,
            'missing': [],
            'extra': [],
            'order_correct': True,
            'direction_match': not auto_reversed,
            'auto_reversed': auto_reversed
        }
    
    # Jeśli pozycyjna dokładność niska, sprawdź zawartość (ignorując kolejność)
    matches = []
    missing = []
    extra = working_detected.copy()
    
    # Sprawdź każdą prawdziwą wartość
    for true_val in true_float:
        found_match = False
        for i, det_val in enumerate(extra):
            if isinstance(true_val, (int, float)) and isinstance(det_val, (int, float)):
                # Porównanie numeryczne z tolerancją
                if abs(true_val - det_val) <= tolerance * max(abs(true_val), abs(det_val), 1):
                    matches.append((true_val, det_val))
                    extra.pop(i)
                    found_match = True
                    break
            else:
                # Porównanie tekstowe
                if str(true_val) == str(det_val):
                    matches.append((true_val, det_val))
                    extra.pop(i)
                    found_match = True
                    break
        
        if not found_match:
            missing.append(true_val)
    
    # Oblicz dokładność zawartości
    content_accuracy = len(matches) / len(true_values) if true_values else 0.0
    
    # Kombinuj dokładność pozycyjną i zawartościową
    # Pozycyjna ma większą wagę (70%), zawartościowa mniejszą (30%)
    final_accuracy = position_accuracy * POSITIONAL_WEIGHT + content_accuracy * CONTENT_WEIGHT
    
    return {
        'accuracy': final_accuracy,
        'detected_count': len(detected_values),
        'true_count': len(true_values),
        'matches': matches,
        'missing': missing,
        'extra': extra,
        'order_correct': position_accuracy >= HIGH_ACCURACY_THRESHOLD,
        'direction_match': not auto_reversed,
        'auto_reversed': auto_reversed
    }

def format_values_display(values, max_display=6):
    """
    Formatuje listę wartości do czytelnego wyświetlania.
    
    Args:
        values: Lista wartości do wyświetlenia
        max_display: Maksymalna liczba wartości do pełnego wyświetlenia
        
    Returns:
        String z sformatowanymi wartościami
    """
    if not values:
        return "brak"
    
    if len(values) <= max_display:
        # Pokaż wszystkie wartości jeśli jest ich mało
        return ', '.join(str(v) for v in values)
    else:
        # Pokaż pierwszą, ostatnią i liczebność
        first_vals = ', '.join(str(v) for v in values[:3])
        last_vals = ', '.join(str(v) for v in values[-2:])
        total = len(values)
        return f"{first_vals} ... {last_vals} (łącznie {total} wartości)"

def format_numeric_range(range_min, range_max):
    """
    Formatuje zakres liczbowy w czytelny sposób.
    
    Args:
        range_min: Minimalna wartość zakresu
        range_max: Maksymalna wartość zakresu
        
    Returns:
        str: Sformatowany zakres w postaci "min ÷ max"
    """
    # Formatowanie dla różnych skal
    def format_number(num):
        if abs(num) >= 1000000:
            return f"{num:.1e}"
        elif abs(num) >= 1000:
            return f"{num:.0f}"
        elif abs(num) >= 1:
            return f"{num:.2g}"
        else:
            return f"{num:.3g}"
    
    return f"{format_number(range_min)} ÷ {format_number(range_max)}"

def format_axis_info(axis_data, true_values=None):
    """
    Formatuje informacje o pojedynczej osi do czytelnego tekstu.
    
    Args:
        axis_data: Słownik z danymi osi
        true_values: Lista prawdziwych wartości (opcjonalnie)
        
    Returns:
        Lista stringów z informacjami o osi
    """
    lines = []
    
    if axis_data['status'] == 'success':
        # Wyświetl wartości - preferuj przekonwertowane jeśli dostępne
        if 'values' in axis_data and axis_data['values'] and 'text_values' in axis_data:
            # Użyj przekonwertowanych wartości liczbowych dla wyświetlania jeśli są różne od text_values
            if len(axis_data['values']) == len(axis_data['text_values']):
                # Sprawdź czy są różnice w konwersji
                display_values = []
                for i, (text_val, num_val) in enumerate(zip(axis_data['text_values'], axis_data['values'])):
                    try:
                        # Jeśli text zawiera operatory lub jest różny od liczby, użyj liczby
                        if any(op in str(text_val) for op in ['>', '<', '≥', '≤', '=']) or str(text_val) != str(num_val):
                            display_values.append(str(num_val))
                        else:
                            display_values.append(str(text_val))
                    except:
                        display_values.append(str(text_val))
                values_str = ', '.join(display_values)
            else:
                values_str = ', '.join(str(v) for v in axis_data['text_values'])
            lines.append(f"        {axis_data['axis_id']}: {values_str}")
        elif 'text_values' in axis_data and axis_data['text_values']:
            values_str = ', '.join(str(v) for v in axis_data['text_values'])
            lines.append(f"        {axis_data['axis_id']}: {values_str}")
            
            # Dodaj porównanie jeśli dostępne prawdziwe wartości
            if true_values is not None:
                # Użyj bezpośrednio przekonwertowanych wartości numerycznych jeśli dostępne
                if 'values' in axis_data and axis_data['values']:
                    detected_numeric = axis_data['values']
                else:
                    # Fallback - konwertuj wykryte wartości do numerycznych jeśli możliwe
                    detected_numeric = []
                    for val_str in axis_data['text_values']:
                        try:
                            # Obsłuż notację potęgową
                            if '^' in str(val_str):
                                base, exp = str(val_str).split('^')
                                detected_numeric.append(float(int(base) ** int(exp)))
                            elif 'E' in str(val_str).upper():
                                detected_numeric.append(float(str(val_str).replace(',', '.')))
                            else:
                                detected_numeric.append(float(val_str))
                        except (ValueError, TypeError):
                            detected_numeric.append(val_str)
                
                comparison = compare_axes_values(detected_numeric, true_values)
                accuracy_pct = comparison['accuracy'] * 100
                
                if accuracy_pct >= 80:
                    status_icon = "✅"
                elif accuracy_pct >= 50:
                    status_icon = "⚠️"
                else:
                    status_icon = "❌"
                
                lines.append(f"            {status_icon} Dokładność: {accuracy_pct:.1f}% ({len(comparison['matches'])}/{comparison['true_count']} wartości)")
                
                # Dodaj informacje o kierunku osi i automatycznym odwracaniu
                if 'auto_reversed' in comparison and comparison['auto_reversed']:
                    lines.append(f"            🔄 Automatycznie odwrócono kolejność (wykryto odwrotny kierunek)")
                elif 'direction_match' in comparison and not comparison['direction_match']:
                    lines.append(f"            ⚠️  Kierunek osi nie pasuje do oczekiwanego")
                
                if comparison['missing']:
                    missing_str = ', '.join(str(v) for v in comparison['missing'])
                    lines.append(f"            Brakuje: {missing_str}")
                
                if comparison['extra']:
                    extra_str = ', '.join(str(v) for v in comparison['extra'])
                    lines.append(f"            Dodatkowe: {extra_str}")
        else:
            lines.append(f"        {axis_data['axis_id']}: brak wartości")
            if true_values:
                lines.append(f"            ❌ Brakuje wszystkich {len(true_values)} wartości")
    else:
        # Błąd interpretacji - pokaż co wykryto
        if 'values' in axis_data and axis_data['values']:
            values_str = ', '.join(str(v) for v in axis_data['values'])
            lines.append(f"        {axis_data['axis_id']}: {values_str}")
        else:
            lines.append(f"        {axis_data['axis_id']}: błąd")
        
        if true_values:
            lines.append(f"            ❌ Błąd interpretacji (oczekiwano {len(true_values)} wartości)")
    
    return lines

def analyze_image_axes(interpretation_file):
    """
    Analizuje plik interpretacji osi dla pojedynczego obrazu.
    
    Args:
        interpretation_file: Ścieżka do pliku JSON z interpretacją osi
        
    Returns:
        Słownik z informacjami o osiach obrazu
    """
    try:
        with open(interpretation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Wyodrębnij nazwę obrazu z nazwy pliku
        filename = os.path.basename(interpretation_file)
        image_name = filename.replace('_axes_interpretation.json', '')
        
        # Policz osie
        horizontal_axes = data.get('horizontal_axes', [])
        vertical_axes = data.get('vertical_axes', [])
        
        horizontal_count = len(horizontal_axes)
        vertical_count = len(vertical_axes)
        
        # Policz pomyślnie zinterpretowane osie
        successful_horizontal = len([ax for ax in horizontal_axes if ax.get('status') == 'success'])
        successful_vertical = len([ax for ax in vertical_axes if ax.get('status') == 'success'])
        
        return {
            'image_name': image_name,
            'file_path': interpretation_file,
            'horizontal_axes': horizontal_axes,
            'vertical_axes': vertical_axes,
            'counts': {
                'horizontal_total': horizontal_count,
                'horizontal_successful': successful_horizontal,
                'vertical_total': vertical_count,
                'vertical_successful': successful_vertical,
                'total_axes': horizontal_count + vertical_count,
                'total_successful': successful_horizontal + successful_vertical
            },
            'has_data': horizontal_count > 0 or vertical_count > 0
        }
        
    except Exception as e:
        return {
            'image_name': os.path.basename(interpretation_file).replace('_axes_interpretation.json', ''),
            'file_path': interpretation_file,
            'error': str(e),
            'has_data': False
        }

def generate_axes_report(results_dir="results/axes", output_file=None, ground_truth_file="charts_axes_detect/true_axes.txt", verbose=False):
    """
    Generuje kompletny raport wszystkich wykrytych osi z porównaniem do prawdziwych wartości.
    
    Args:
        results_dir: Katalog z wynikami interpretacji osi
        output_file: Opcjonalna ścieżka do zapisania raportu do pliku
        ground_truth_file: Ścieżka do pliku z prawdziwymi wartościami
        verbose: Czy pokazywać szczegółowe porównania dla każdego obrazu
        
    Returns:
        String z raportem
    """
    print("Skanowanie plików interpretacji osi...")
    
    # Wczytaj prawdziwe wartości
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"📊 Wczytano prawdziwe wartości dla {len(ground_truth)} obrazów")
    
    # Znajdź wszystkie pliki interpretacji
    pattern = os.path.join(results_dir, "*_axes_interpretation.json")
    interpretation_files = glob.glob(pattern)
    
    if not interpretation_files:
        return f"Nie znaleziono plików interpretacji osi w katalogu: {results_dir}"
    
    print(f"Znaleziono {len(interpretation_files)} plików interpretacji")
    
    # Analizuj każdy plik
    image_analyses = []
    total_comparisons = 0
    total_x_accuracy = 0.0
    total_y_accuracy = 0.0
    x_comparisons = 0
    y_comparisons = 0
    
    for file_path in sorted(interpretation_files):
        analysis = analyze_image_axes(file_path)
        image_analyses.append(analysis)
    
    # Generuj raport
    report_lines = []
    
    # Szczegóły dla każdego obrazu
    for img_data in image_analyses:
        if 'error' in img_data:
            continue
        
        image_name = img_data['image_name']
        
        if not img_data['has_data']:
            continue
        
        # Nazwa pliku
        report_lines.append(f"{image_name}")
        
        # Pobierz prawdziwe wartości dla tego obrazu
        true_data = ground_truth.get(image_name, {})
        true_x = true_data.get('X', [])
        true_y = true_data.get('Y', [])
        
        # Osie X
        if img_data['horizontal_axes']:
            report_lines.append("    OSIE X:")
            for axis in img_data['horizontal_axes']:
                axis_lines = format_axis_info(axis, true_x)
                report_lines.extend(axis_lines)
                
                # Dodaj do statystyk globalnych
                if true_x and axis.get('status') == 'success' and axis.get('values'):
                    # Użyj bezpośrednio przekonwertowanych wartości numerycznych
                    detected_numeric = axis['values']
                    
                    comparison = compare_axes_values(detected_numeric, true_x)
                    total_x_accuracy += comparison['accuracy']
                    x_comparisons += 1
                    
                    # Verbose logging dla osi X
                    if verbose:
                        accuracy_percent = comparison['accuracy'] * 100
                        report_lines.append(f"        🔍 PORÓWNANIE X: {accuracy_percent:.1f}%")
                        report_lines.append(f"        Prawdziwe: {true_x}")
                        report_lines.append(f"        Wykryte:   {detected_numeric}")
                        if comparison['missing']:
                            report_lines.append(f"        Brakuje:   {comparison['missing']}")
                        if comparison['extra']:
                            report_lines.append(f"        Nadmiar:   {comparison['extra']}")
                        if comparison['auto_reversed']:
                            report_lines.append(f"        🔄 Automatycznie odwrócono kolejność")
                        report_lines.append("")
        
        # Osie Y  
        if img_data['vertical_axes']:
            report_lines.append("    OSIE Y:")
            for axis in img_data['vertical_axes']:
                axis_lines = format_axis_info(axis, true_y)
                report_lines.extend(axis_lines)
                
                # Dodaj do statystyk globalnych
                if true_y and axis.get('status') == 'success' and axis.get('values'):
                    # Użyj bezpośrednio przekonwertowanych wartości numerycznych
                    detected_numeric = axis['values']
                    
                    comparison = compare_axes_values(detected_numeric, true_y)
                    total_y_accuracy += comparison['accuracy']
                    y_comparisons += 1
                    
                    # Verbose logging dla osi Y
                    if verbose:
                        accuracy_percent = comparison['accuracy'] * 100
                        report_lines.append(f"        🔍 PORÓWNANIE Y: {accuracy_percent:.1f}%")
                        report_lines.append(f"        Prawdziwe: {true_y}")
                        report_lines.append(f"        Wykryte:   {detected_numeric}")
                        if comparison['missing']:
                            report_lines.append(f"        Brakuje:   {comparison['missing']}")
                        if comparison['extra']:
                            report_lines.append(f"        Nadmiar:   {comparison['extra']}")
                        if comparison['auto_reversed']:
                            report_lines.append(f"        🔄 Automatycznie odwrócono kolejność")
                        report_lines.append("")
        
        report_lines.append("")
    
    # Dodaj statystyki globalne na końcu
    report_lines.append("=" * 60)
    report_lines.append("PODSUMOWANIE DOKŁADNOŚCI")
    report_lines.append("=" * 60)
    
    if x_comparisons > 0:
        avg_x_accuracy = (total_x_accuracy / x_comparisons) * 100
        report_lines.append(f"📊 Średnia dokładność osi X: {avg_x_accuracy:.1f}% ({x_comparisons} porównań)")
    else:
        report_lines.append("📊 Brak porównań dla osi X")
    
    if y_comparisons > 0:
        avg_y_accuracy = (total_y_accuracy / y_comparisons) * 100
        report_lines.append(f"📊 Średnia dokładność osi Y: {avg_y_accuracy:.1f}% ({y_comparisons} porównań)")
    else:
        report_lines.append("📊 Brak porównań dla osi Y")
    
    total_comparisons = x_comparisons + y_comparisons
    if total_comparisons > 0:
        overall_accuracy = ((total_x_accuracy + total_y_accuracy) / total_comparisons) * 100
        report_lines.append(f"🎯 Ogólna dokładność: {overall_accuracy:.1f}% ({total_comparisons} osi łącznie)")
    
    report_lines.append("")
    report_lines.append(f"📁 Analizowano {len([img for img in image_analyses if img.get('has_data')])} obrazów z osiami")
    report_lines.append(f"📋 Dostępne prawdziwe wartości dla {len(ground_truth)} obrazów")
    
    # Złącz linie w jeden string
    report = "\n".join(report_lines)
    
    # Zapisz do pliku jeśli podano ścieżkę
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Raport zapisano do: {output_file}")
        except Exception as e:
            print(f"Błąd zapisu raportu: {e}")
    
    return report

def main():
    """Główna funkcja uruchamiająca generator raportu."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generator raportu wykrytych osi na wykresach z porównaniem do prawdziwych wartości",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--results-dir", "-d", 
                       default="results/axes",
                       help="Katalog z plikami interpretacji osi")
    
    parser.add_argument("--output", "-o", 
                       help="Plik do zapisania raportu (opcjonalnie)")
    
    parser.add_argument("--ground-truth", "-g",
                       default="charts_axes_detect/true_axes.txt", 
                       help="Plik z prawdziwymi wartościami osi")
    
    parser.add_argument("--print", "-p", 
                       action="store_true",
                       help="Wypisz raport na konsoli")
    
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Pokazuj szczegółowe porównania dla każdego obrazu")
    
    args = parser.parse_args()
    
    # Sprawdź czy katalog istnieje
    if not os.path.exists(args.results_dir):
        print(f"❌ Katalog nie istnieje: {args.results_dir}")
        return
    
    # Generuj raport
    print("🚀 Uruchamiam generator raportu osi z porównaniem...")
    report = generate_axes_report(args.results_dir, args.output, args.ground_truth, verbose=args.verbose)
    
    # Wypisz raport jeśli requested
    if args.print or not args.output:
        print("\n" + report)

if __name__ == "__main__":
    main() 