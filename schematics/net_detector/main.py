import cv2
import numpy as np
from line_detector import LineDetector
import json
import os
import logging
import sys
import traceback
from datetime import datetime

# Konfiguracja loggera z zapisem do pliku
log_file = 'program_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Dodaj funkcję wypisywania błędów
def log_error(message, exception=None):
    logging.error(message)
    print(f"BŁĄD: {message}")
    if exception:
        logging.error(f"Szczegóły błędu: {str(exception)}")
        logging.error(traceback.format_exc())

def process_single_schema(filename, schema_type):
    """Przetwarza pojedynczy schemat."""
    try:
        print(f"Przetwarzam schemat: {filename} typu: {schema_type}")
        logging.info(f"Rozpoczynam przetwarzanie schematu: {filename} typu: {schema_type}")
        
        # Inicjalizacja detektora linii
        line_detector = LineDetector()
        logging.info(f"Zainicjalizowano LineDetector dla {filename}")
        
        # Wczytaj adnotacje z katalogu combined_json
        annotations_dir = os.path.join('combined_json', schema_type)
        annotations_path = os.path.join(annotations_dir, filename)
        print(f"Próba wczytania adnotacji z: {annotations_path}")
        logging.info(f"Próba wczytania adnotacji z: {annotations_path}")
        
        if not os.path.exists(annotations_path):
            log_error(f"Nie znaleziono pliku adnotacji: {annotations_path}")
            return None
            
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Wczytano adnotacje dla {filename}")
        
        # Wczytaj oryginalny obraz w pełnej rozdzielczości
        image_path = data['image_path']
        print(f"Próba wczytania obrazu z: {image_path}")
        logging.info(f"Próba wczytania obrazu z: {image_path}")
        
        if not os.path.exists(image_path):
            log_error(f"Nie znaleziono pliku obrazu: {image_path}")
            return None
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            log_error(f"Nie udało się wczytać obrazu: {image_path}")
            return None
        logging.info(f"Wczytano oryginalny obraz: {image_path}")
        
        # Pobierz wymiary obrazu z adnotacji
        original_width = data.get('image_size', {}).get('width', 800)
        original_height = data.get('image_size', {}).get('height', 600)
        
        # Oblicz skalę na podstawie wymiarów obrazu
        scale_x = original_image.shape[1] / original_width
        scale_y = original_image.shape[0] / original_height
        scale = min(scale_x, scale_y)  # Używamy mniejszej skali, aby zachować proporcje
        
        # Przeskaluj współrzędne bloków
        scaled_blocks = []
        for block in data['blocks']:
            coords = block['coords']
            # Skaluj współrzędne używając obliczonej skali
            scaled_coords = [
                float(coords[0]) * scale,
                float(coords[1]) * scale,
                float(coords[2]) * scale,
                float(coords[3]) * scale
            ]
            scaled_blocks.append({'coords': scaled_coords})
            
        print(f"Przygotowano {len(scaled_blocks)} przeskalowanych bloków")
        logging.info(f"Przygotowano {len(scaled_blocks)} przeskalowanych bloków")
        
        # Przygotuj katalogi wynikowe
        output_dir = os.path.join('results', schema_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Przygotuj katalog debug dla tego schematu
        filename_base = os.path.splitext(os.path.basename(filename))[0]
        debug_dir = os.path.join('results', 'debug', schema_type, filename_base)
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Utworzono katalog debug: {debug_dir}")
        logging.info(f"Utworzono katalog debug: {debug_dir}")
        
        # Wykryj linie na obrazie, przekazując ścieżkę do katalogu debug
        print(f"Rozpoczynam wykrywanie linii...")
        connections = line_detector.detect_lines(original_image, scaled_blocks, debug_dir)
        print(f"Wykryto {len(connections)} połączeń w {filename}")
        logging.info(f"Wykryto {len(connections)} połączeń w {filename}")
        
        # Zapisz wynikowy obraz
        line_detector.save_result_image(output_dir, filename_base)
        print(f"Zapisano wynikowy obraz do: {output_dir}")
        logging.info(f"Zapisano wynikowy obraz do: {output_dir}")
        
        return connections
        
    except Exception as e:
        log_error(f"Błąd podczas przetwarzania schematu {filename}", e)
        return None

def process_schemas(schema_type):
    """Przetwarza wszystkie schematy danego typu."""
    try:
        print(f"Rozpoczynam przetwarzanie schematów typu: {schema_type}")
        logging.info(f"Rozpoczynam przetwarzanie schematów typu: {schema_type}")
        
        # Katalog z adnotacjami z combined_json
        annotations_dir = os.path.join('combined_json', schema_type)
        print(f"Sprawdzam katalog: {annotations_dir}")
        if not os.path.exists(annotations_dir):
            log_error(f"Nie znaleziono katalogu z adnotacjami: {annotations_dir}")
            return
        
        # Pobierz listę plików JSON z adnotacjami
        try:
            json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
            print(f"Znaleziono {len(json_files)} plików z adnotacjami w {annotations_dir}")
            logging.info(f"Znaleziono {len(json_files)} plików z adnotacjami w {annotations_dir}")
        except Exception as e:
            log_error(f"Błąd podczas listowania plików w katalogu {annotations_dir}", e)
            return
        
        # Przetwórz każdy plik z adnotacjami
        for json_file in json_files:
            process_single_schema(json_file, schema_type)
        
        print(f"Zakończono przetwarzanie schematów typu: {schema_type}")
        logging.info(f"Zakończono przetwarzanie schematów typu: {schema_type}")
    except Exception as e:
        log_error(f"Błąd w process_schemas dla typu {schema_type}", e)

def main():
    try:
        print("Uruchamiam program wykrywania linii...")
        logging.info("Uruchamiam program wykrywania linii...")
        
        # Sprawdź czy katalogi istnieją
        if not os.path.exists('combined_json'):
            log_error("Nie znaleziono katalogu combined_json")
            input("Naciśnij Enter, aby zamknąć program...")
            return
            
        print("Sprawdzam strukturę katalogów...")
        # Sprawdź podfoldery
        for schema_type in ['Automatyka', 'Elektroniczne']:
            dir_path = os.path.join('combined_json', schema_type)
            if not os.path.exists(dir_path):
                log_error(f"Nie znaleziono katalogu combined_json/{schema_type}")
                input("Naciśnij Enter, aby zamknąć program...")
                return
        
        # Utwórz katalogi wyników
        print("Tworzę katalogi wynikowe...")
        os.makedirs(os.path.join('results', 'Automatyka'), exist_ok=True)
        os.makedirs(os.path.join('results', 'Elektroniczne'), exist_ok=True)
        
        # Utwórz katalogi debug
        print("Tworzę katalogi debug...")
        os.makedirs(os.path.join('results', 'debug', 'Automatyka'), exist_ok=True)
        os.makedirs(os.path.join('results', 'debug', 'Elektroniczne'), exist_ok=True)
        print("Utworzono katalogi debug")
        logging.info("Utworzono katalogi debug")
        
        # Przetwórz schematy
        process_schemas('Automatyka')
        process_schemas('Elektroniczne')
        
        print("Zakończono przetwarzanie wszystkich schematów")
        logging.info("Zakończono przetwarzanie wszystkich schematów")
    except Exception as e:
        log_error("Błąd główny", e)
    finally:
        print(f"\nLogi zapisano do pliku: {log_file}")
        print(f"Program zakończył działanie.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error("Nieprzewidziany błąd", e)
    finally:
        input("Naciśnij Enter, aby zamknąć program...") 