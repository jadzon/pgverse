#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from .chart_preprocessing import preprocess_for_small_text
from .small_text_ocr import detect_text_combined
from .run_text_extraction import run_chart_text_extraction, format_results_to_json, annotate_image
from .axes_detection import process_results_for_axes, process_image_for_axes
from .axes_interpretation import process_axes_interpretation

# Konfiguracja - ustawienia domyślne
CONFIG = {
    # Katalogi
    "input_directory": "../charts_examples",
    "output_directory": "results",
    "preprocessed_directory": "preprocessed_charts",
    
    # Ustawienia wykrywania tekstu
    "min_confidence": 0.1,
    "enable_merging": True,
    "iou_threshold": 0.4,
    "save_intermediate": False,
    
    # Ustawienia wykrywania osi
    "detect_axes": True,
    "extension_factor": 5.0,
    "min_extension": 75,
    "overlap_threshold": 0.01,
    "alignment_tolerance": 0.5,
    
    # Ustawienia interpretacji osi
    "interpret_axes": True,
    
    # Tryb pracy
    "single_file": None,  # None - przetwarzaj cały katalog, ścieżka - przetwarzaj jeden plik
}

def process_single_file(file_path, output_dir, config):
    """
    Przetwarzanie pojedynczego pliku obrazu bez etapu preprocessingu.
    
    Args:
        file_path: Ścieżka do pliku obrazu
        output_dir: Katalog wyjściowy
        config: Słownik konfiguracyjny
        
    Returns:
        Dict zawierający dane o wykrytym tekście i osiach
    """
    if not os.path.exists(file_path):
        print(f"Błąd: Plik {file_path} nie istnieje")
        return None
    
    # Upewnij się, że katalog wyjściowy istnieje
    os.makedirs(output_dir, exist_ok=True)
    
    # Utwórz podkatalog dla wyników wykrywania osi
    axes_output_dir = os.path.join(output_dir, "axes")
    os.makedirs(axes_output_dir, exist_ok=True)
    
    # Uruchom detekcję tekstu
    print(f"\nPrzetwarzanie {file_path}...")
    results = detect_text_combined(
        file_path,
        min_confidence=config["min_confidence"],
        enable_merging=config["enable_merging"],
        iou_merge_threshold=config["iou_threshold"],
        return_intermediate=config["save_intermediate"]
    )
    
    # Przetwórz wyniki
    import cv2
    import json
    
    # Przygotuj nazwę bazową pliku
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    
    # Wczytaj obraz do adnotacji
    img = cv2.imread(file_path)
    if img is None:
        print(f"  Błąd: Nie można odczytać obrazu {file_path}")
        return None
    
    if config["save_intermediate"]:
        # Zapisz wyniki pośrednie
        for stage, detection_list in results.items():
            if detection_list:
                annotated = annotate_image(img, detection_list)
                stage_path = os.path.join(output_dir, f"{name}_{stage}.png")
                cv2.imwrite(stage_path, annotated)
        
        # Wyodrębnij wyniki końcowe
        final_detections = results['final']
    else:
        # W tym przypadku results to bezpośrednio lista detekcji
        final_detections = results
    
    # Zapisz wyniki końcowe
    json_data = None
    axes_data = None
    
    if final_detections:
        # Adnotowany obraz z końcowymi detekcjami
        annotated_final = annotate_image(img, final_detections)
        final_path = os.path.join(output_dir, f"{name}_final.png")
        cv2.imwrite(final_path, annotated_final)
        
        # Dane JSON
        json_data = format_results_to_json(file_path, final_detections, img.shape)
        json_path = os.path.join(output_dir, f"{name}_text.json")
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"  Zapisano wyniki tekstu do: {json_path}")
        except Exception as e:
            print(f"  Błąd podczas zapisywania JSON: {e}")
        
        print(f"  Wykryto {len(final_detections)} bloków tekstowych")
        
        # Wykryj osie na podstawie bloków tekstu
        if config["detect_axes"] and len(final_detections) >= 2:
            print(f"\nWykrywanie osi dla {file_path}...")
            
            # Wykryj osie bezpośrednio z wykrytych bloków tekstu
            axes_data = process_image_for_axes(
                file_path,
                final_detections,
                axes_output_dir,
                config["extension_factor"],
                config["min_extension"],
                config["overlap_threshold"],
                config["alignment_tolerance"]
            )
            
            if axes_data:
                horizontal_count = len(axes_data["horizontal_axes"])
                vertical_count = len(axes_data["vertical_axes"])
                print(f"  Wykryto {horizontal_count} osi poziomych i {vertical_count} osi pionowych")
                print(f"  Zapisano wyniki osi w katalogu: {axes_output_dir}")
                
                # Interpretuj osie, jeśli włączono tę opcję
                if config["interpret_axes"] and (horizontal_count > 0 or vertical_count > 0):
                    print(f"\nInterpretacja osi dla {file_path}...")
                    axes_json_path = os.path.join(axes_output_dir, f"{name}_axes.json")
                    
                    interpretation = process_axes_interpretation(axes_json_path, axes_output_dir)
    else:
        print(f"  Nie wykryto żadnego tekstu w {file_path}")
    
    return interpretation,axes_data

def main():
    """Główna funkcja programu"""
    # Pobierz wartości z konfiguracji
    input_dir = CONFIG["input_directory"]
    output_dir = CONFIG["output_directory"]
    preprocessed_dir = CONFIG["preprocessed_directory"]
    single_file = CONFIG["single_file"]
    
    print("=== Chart Text Detection ===")
    print(f"Poziom pewności: {CONFIG['min_confidence']}")
    print(f"Łączenie bloków: {'Wyłączone' if not CONFIG['enable_merging'] else 'Włączone'}")
    if CONFIG["enable_merging"]:
        print(f"Próg IoU: {CONFIG['iou_threshold']}")
    
    if CONFIG["detect_axes"]:
        print("\n=== Wykrywanie Osi ===")
        print(f"Współczynnik rozszerzenia: {CONFIG['extension_factor']}")
        print(f"Minimalne rozszerzenie: {CONFIG['min_extension']} px")
        print(f"Próg nakładania: {CONFIG['overlap_threshold']}")
        print(f"Tolerancja wyrównania: {CONFIG['alignment_tolerance']}")
    
    if CONFIG["interpret_axes"]:
        print("\n=== Interpretacja Osi ===")
        print(f"Interpretacja osi: {'Włączona' if CONFIG['interpret_axes'] else 'Wyłączona'}")
    
    # Jeśli podano pojedynczy plik, przetwarzamy tylko jego
    if single_file:
        print(f"Tryb pojedynczego pliku: {single_file}")
        result = process_single_file(
            single_file, 
            output_dir, 
            CONFIG
        )
        if result:
            print("Przetwarzanie zakończone pomyślnie.")
        else:
            print("Przetwarzanie zakończone z błędami.")
        return
    
    # W przeciwnym razie przetwarzamy cały katalog
    print(f"Katalog wejściowy: {input_dir}")
    print(f"Katalog wyjściowy: {output_dir}")
    print(f"Katalog przetworzonych obrazów: {preprocessed_dir}")
    
    # Uruchom pełny proces OCR
    results = run_chart_text_extraction(
        original_input_folder=input_dir,
        preprocessed_folder=preprocessed_dir,
        output_folder=output_dir
    )
    
    # Utwórz podkatalog dla wyników wykrywania osi
    axes_output_dir = os.path.join(output_dir, "axes")
    os.makedirs(axes_output_dir, exist_ok=True)
    
    # Teraz wykonaj wykrywanie osi dla wszystkich przetworzonych plików
    if CONFIG["detect_axes"] and results:
        print("\n=== Wykrywanie Osi ===")
        
        axes_results = {}
        for image_name, json_data in results.items():
            json_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_text.json")
            
            if os.path.exists(json_path):
                print(f"\nWykrywanie osi dla {image_name}...")
                
                # Przetwórz plik JSON z wynikami OCR
                axes_data = process_results_for_axes(
                    json_path,
                    axes_output_dir,
                    CONFIG["extension_factor"],
                    CONFIG["min_extension"],
                    CONFIG["overlap_threshold"],
                    CONFIG["alignment_tolerance"]
                )
                
                if axes_data:
                    axes_results[image_name] = axes_data
                    
                    # Interpretuj osie, jeśli włączono tę opcję
                    if CONFIG["interpret_axes"]:
                        print(f"\nInterpretacja osi dla {image_name}...")
                        axes_json_path = os.path.join(axes_output_dir, f"{os.path.splitext(image_name)[0]}_axes.json")
                        
                        if os.path.exists(axes_json_path):
                            interpretation = process_axes_interpretation(axes_json_path, axes_output_dir)
    
    # Podsumowanie
    print("\n=== Podsumowanie ===")
    print(f"Przetworzono {len(results)} obrazów.")
    
    # Zlicz całkowitą liczbę bloków tekstu
    total_blocks = sum(len(data.get("blocks", [])) for data in results.values())
    print(f"Całkowita liczba wykrytych bloków tekstu: {total_blocks}")
    
    if CONFIG["detect_axes"] and 'axes_results' in locals() and axes_results:
        total_horizontal_axes = sum(len(data.get("horizontal_axes", [])) for data in axes_results.values())
        total_vertical_axes = sum(len(data.get("vertical_axes", [])) for data in axes_results.values())
        print(f"Wykryto {total_horizontal_axes} osi poziomych i {total_vertical_axes} osi pionowych")
        print(f"Wyniki wykrywania osi zapisano w katalogu: {axes_output_dir}")
    
    print("\nPrzetwarzanie zakończone pomyślnie.")

if __name__ == "__main__":
    main() 