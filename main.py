import cv2
import numpy as np
from connection_detector import ConnectionDetector
import json
import os
import hashlib
import pickle
import logging
from datetime import datetime

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_cache_path(image_path, schema_type):
    """Generuje ścieżkę do pliku cache."""
    # Użyj nazwy pliku zamiast pełnej ścieżki
    filename = os.path.basename(image_path)
    hash_object = hashlib.md5(filename.encode())
    return os.path.join('cache', schema_type, f"{hash_object.hexdigest()}.pkl")

def process_single_schema(filename, schema_type, sequence_number):
    """Przetwarza pojedynczy schemat."""
    cache_path = get_cache_path(filename, schema_type)
    connections = None
    
    try:
        logging.info(f"Rozpoczynam przetwarzanie schematu: {filename} typu: {schema_type}")
        
        # Sprawdź czy wyniki są w cache
        if os.path.exists(cache_path):
            logging.info(f"Znaleziono plik w cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                connections = pickle.load(f)
                logging.info(f"Wykryto {len(connections)} połączeń w {filename}")
        else:
            logging.info(f"Nie znaleziono pliku w cache: {cache_path}")
        
        # Inicjalizacja detektora połączeń
        connection_detector = ConnectionDetector()
        logging.info(f"Zainicjalizowano ConnectionDetector dla {filename}")
        
        # Wczytaj adnotacje
        annotations_dir = os.path.join('annotations', schema_type)
        annotations_path = os.path.join(annotations_dir, filename)
        logging.info(f"Próba wczytania adnotacji z: {annotations_path}")
        
        if not os.path.exists(annotations_path):
            logging.error(f"Nie znaleziono pliku adnotacji: {annotations_path}")
            return None
            
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Wczytano adnotacje dla {filename}")
        
        # Wczytaj oryginalny obraz w pełnej rozdzielczości
        image_path = data['image_path']
        logging.info(f"Próba wczytania obrazu z: {image_path}")
        
        if not os.path.exists(image_path):
            logging.error(f"Nie znaleziono pliku obrazu: {image_path}")
            return None
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            logging.error(f"Nie udało się wczytać obrazu: {image_path}")
            return None
        logging.info(f"Wczytano oryginalny obraz: {image_path}")
        
        # Zapisz oryginalny obraz do debug_images
        connection_detector.debug_images['original'] = original_image.copy()
        logging.info(f"Dodano oryginalny obraz do debug_images dla {filename}")
        
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
            
        logging.info(f"Przygotowano {len(scaled_blocks)} przeskalowanych bloków")
        logging.info(f"Wymiary oryginalnego obrazu: {original_image.shape}")
        logging.info(f"Wymiary z adnotacji: {original_width}x{original_height}")
        logging.info(f"Użyta skala: {scale}")
        logging.info(f"Przykładowe przeskalowane współrzędne: {scaled_blocks[0]['coords'] if scaled_blocks else 'brak bloków'}")
        
        # Przetwórz obraz
        logging.info(f"Rozpoczynam przetwarzanie obrazu dla {filename}")
        processed = connection_detector.preprocess_for_lines(original_image)
        if processed is None:
            logging.error(f"Przetwarzanie obrazu nie powiodło się dla {filename}")
            return None
        logging.info(f"Zakończono przetwarzanie obrazu dla {filename}")
        
        # Wykryj połączenia na obrazie (tylko jeśli nie ma w cache)
        if connections is None:
            logging.info(f"Rozpoczynam wykrywanie połączeń dla {filename}")
            connections = connection_detector.detect_connections(processed, scaled_blocks)
            logging.info(f"Wykryto {len(connections)} połączeń w {filename}")
        
        # Zapisz obrazy debugowe
        debug_dir = os.path.join('results', 'debug', schema_type)
        os.makedirs(debug_dir, exist_ok=True)
        logging.info(f"Liczba dostępnych obrazów debugowych: {len(connection_detector.debug_images)}")
        logging.info(f"Zapisywanie obrazów debugowych do: {debug_dir}")
        logging.info(f"Lista dostępnych etapów debugowania: {list(connection_detector.debug_images.keys())}")
        
        try:
            # Dodaj numer sekwencyjny do nazwy pliku
            base_name = os.path.splitext(filename)[0]
            debug_filename = f"{sequence_number:03d}_{base_name}"
            logging.info(f"Rozpoczynam zapisywanie obrazów debugowych dla {debug_filename}")
            connection_detector.save_debug_images(debug_dir, debug_filename)
            logging.info(f"Pomyślnie zapisano obrazy debugowe dla {debug_filename}")
            
            # Zapisz finalną wersję obrazu z narysowanymi liniami
            results_dir = os.path.join('results', schema_type)
            os.makedirs(results_dir, exist_ok=True)
            final_filename = f"{sequence_number:03d}_{base_name}_final.png"
            final_path = os.path.join(results_dir, final_filename)
            cv2.imwrite(final_path, connection_detector.debug_images['final'])
            logging.info(f"Zapisano finalną wersję obrazu do: {final_path}")
            
        except Exception as e:
            logging.error(f"Błąd podczas zapisywania obrazów dla {filename}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        
        # Zapisz wyniki do cache (tylko jeśli nie ma w cache)
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(connections, f)
            logging.info(f"Zapisano wyniki do cache dla {filename}")
        
        # Zwolnij pamięć
        del processed
        del original_image
        
        return connections
        
    except Exception as e:
        logging.error(f"Błąd podczas przetwarzania {filename}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def process_schemas(schema_type):
    """Przetwarza schematy danego typu."""
    annotations_dir = os.path.join('annotations', schema_type)
    files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    logging.info(f"Rozpoczynam przetwarzanie schematów typu: {schema_type}")
    logging.info(f"Znaleziono {len(files)} plików do przetworzenia")
    
    # Sortuj pliki alfabetycznie, aby zachować spójną kolejność
    files.sort()
    
    for idx, filename in enumerate(files, 1):
        logging.info(f"Przetwarzanie pliku {idx}/{len(files)}: {filename}")
        result = process_single_schema(filename, schema_type, idx)
        if result is None:
            logging.error(f"Przetwarzanie nie powiodło się dla {filename}")
        
        # Wymuszamy zwolnienie pamięci
        import gc
        gc.collect()

def main():
    # Usuń pliki cache
    import shutil
    import time
    
    if os.path.exists('cache'):
        try:
            logging.info("Rozpoczynam czyszczenie katalogu cache...")
            # Najpierw usuń zawartość katalogów
            for schema_type in ['Automatyka', 'Elektroniczne']:
                cache_dir = os.path.join('cache', schema_type)
                if os.path.exists(cache_dir):
                    logging.info(f"Usuwam zawartość katalogu: {cache_dir}")
                    for file in os.listdir(cache_dir):
                        file_path = os.path.join(cache_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            logging.error(f"Błąd podczas usuwania pliku {file_path}: {str(e)}")
            
            # Następnie usuń katalogi
            logging.info("Usuwam katalogi cache...")
            shutil.rmtree('cache')
            logging.info("Pomyślnie usunięto katalog cache")
            
        except PermissionError:
            logging.warning("Nie można usunąć katalogu cache - odmowa dostępu. Spróbuję ponownie za 1 sekundę...")
            time.sleep(1)  # Poczekaj 1 sekundę
            try:
                shutil.rmtree('cache')
                logging.info("Usunięto katalog cache po ponownej próbie")
            except Exception as e:
                logging.error(f"Nie udało się usunąć katalogu cache: {str(e)}")
                logging.info("Kontynuuję bez usuwania cache...")
        except Exception as e:
            logging.error(f"Wystąpił błąd podczas usuwania cache: {str(e)}")
            logging.info("Kontynuuję bez usuwania cache...")
    else:
        logging.info("Katalog cache nie istnieje, nie ma potrzeby czyszczenia")
    
    # Utwórz katalogi na wyniki
    os.makedirs('results/Automatyka', exist_ok=True)
    os.makedirs('results/Elektroniczne', exist_ok=True)
    
    # Utwórz katalog cache
    os.makedirs('cache/Automatyka', exist_ok=True)
    os.makedirs('cache/Elektroniczne', exist_ok=True)
    
    # Utwórz katalog debug
    os.makedirs('results/debug/Automatyka', exist_ok=True)
    os.makedirs('results/debug/Elektroniczne', exist_ok=True)
    
    # Przetwórz schematy automatyki
    process_schemas('Automatyka')
    
    # Przetwórz schematy elektroniczne
    process_schemas('Elektroniczne')

if __name__ == "__main__":
    main() 