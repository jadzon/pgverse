import os
import json
import sys
import time
import gc
import torch
from pathlib import Path

# Dodaj katalog nadrzędny do ścieżki, aby umożliwić import z rag_functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag_functions.captioning import ImageCaptioner

def test_model_loading():
    """
    Sprawdza, czy modele mogą zostać załadowane z bieżącymi ustawieniami pamięci.
    Zwraca True, jeśli test się powiedzie, False w przeciwnym razie.
    """
    try:
        print("Testowanie możliwości załadowania modelu...")
        test_captioner = ImageCaptioner(
            preload_models=False,
            use_gpu=torch.cuda.is_available(),
            caption_model_name="Salesforce/blip2-opt-2.7b"
        )
        test_captioner.cleanup()
        del test_captioner
        print("Test załadowania modelu zakończony powodzeniem.")
        return True
    except Exception as e:
        print(f"Test załadowania modelu nieudany: {e}")
        return False

def create_image_descriptions_json(images_folder, output_json_path=None, use_gpu=True, resume_from=None):
    """
    Przetwarza wszystkie zdjęcia w folderze, generuje opisy po polsku i zapisuje wyniki do pliku JSON.
    Po każdym przetworzonym obrazie zapisuje częściowe wyniki.
    
    Args:
        images_folder (str): Ścieżka do folderu ze zdjęciami
        output_json_path (str, optional): Ścieżka do zapisu pliku JSON. Jeśli None, używa domyślnej lokalizacji
        use_gpu (bool): Czy używać GPU do generowania opisów
        resume_from (int): Od którego obrazu wznowić przetwarzanie. Jeśli None, sprawdza istniejące wyniki
    
    Returns:
        str: Ścieżka do zapisanego pliku JSON
    """
    # Ustawienia dla lepszego zarządzania pamięcią GPU - dostosowane do problemów z pamięcią
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    torch.backends.cuda.matmul.allow_tf32 = True  # przyspiesza obliczenia na nowszych GPU
    
    # Normalizacja ścieżek
    images_folder = os.path.normpath(images_folder)
    
    # Domyślna ścieżka wyjściowa, jeśli nie podano
    if output_json_path is None:
        project_root = Path(__file__).parent.parent
        output_json_path = os.path.join(project_root, "data", "input1", "opisy_zdjec.json")
    
    # Upewnij się, że output_json_path jest ścieżką do pliku, a nie folderem
    if os.path.isdir(output_json_path):
        output_json_path = os.path.join(output_json_path, "opisy_zdjec.json")
    
    output_json_path = os.path.normpath(output_json_path)
    print(f"Plik wyjściowy JSON: {output_json_path}")
    
    # Upewnij się, że katalog wyjściowy istnieje
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Sprawdź, czy istnieją już częściowe wyniki
    results = []
    processed_paths = set()
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_paths = {item["path"] for item in results}
                print(f"Wczytano {len(results)} istniejących opisów z {output_json_path}")
        except Exception as e:
            print(f"Błąd podczas wczytywania istniejącego pliku JSON: {e}")
    
    # Wspierane rozszerzenia plików graficznych
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # Znajdź wszystkie pliki obrazów w folderze
    image_files = []
    for file in os.listdir(images_folder):
        file_path = os.path.join(images_folder, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_path.lower())
            if ext in image_extensions:
                image_files.append(file_path)
    
    print(f"Znaleziono {len(image_files)} plików obrazów w {images_folder}")
    
    # Jeśli podano indeks wznowienia, rozpocznij od tego miejsca
    start_index = 0
    if resume_from is not None:
        start_index = max(0, min(resume_from, len(image_files)))
        print(f"Wznawiam przetwarzanie od obrazu {start_index+1}")
    
    # Licznik udanych opisów
    successful_count = 0
    
    # Jeśli używamy GPU, spróbuj załadować model testowy aby sprawdzić czy pamięć wystarczy
    if use_gpu and not test_model_loading():
        print("UWAGA: Automatycznie przełączam na tryb CPU z powodu problemów z pamięcią GPU.")
        print("Możesz też spróbować:")
        print("1. Zwiększyć rozmiar pliku stronicowania w systemie Windows")
        print("2. Użyj mniejszego modelu (zmień na blip2-flan-t5-xl)")
        use_gpu = False
    
    for i in range(start_index, len(image_files)):
        image_path = image_files[i]
        
        # Pomiń już przetworzone obrazy
        if image_path in processed_paths:
            print(f"Pomijam już przetworzony obraz {i+1}/{len(image_files)}: {image_path}")
            continue
        
        print(f"Przetwarzanie obrazu {i+1}/{len(image_files)}: {image_path}")
        
        # Wymuś czyszczenie pamięci
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        
        # Inicjalizuj captioner dla każdego obrazu, aby uniknąć wycieków pamięci
        success = False
        max_retries = 3
        captioner = None
        
        # Spróbuj przetworzyć obraz kilkukrotnie z różnymi ustawieniami
        for attempt in range(max_retries):
            try:
                # Inicjalizacja nowego captionera za każdym razem
                if captioner is not None:
                    try:
                        captioner.cleanup()
                    except:
                        pass
                    del captioner
                    captioner = None
                    if use_gpu:
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Utwórz nowy captioner z odpowiednimi ustawieniami
                current_use_gpu = use_gpu and (attempt < 1)  # Na ostatnich dwóch próbach użyj CPU
                smaller_model = attempt == 2  # Na ostatniej próbie użyj mniejszego modelu
                
                model_name = "Salesforce/blip2-flan-t5-xl" if smaller_model else "Salesforce/blip2-opt-2.7b"
                print(f"Próba {attempt+1}/{max_retries}: {'GPU' if current_use_gpu else 'CPU'}, model: {model_name}")
                
                captioner = ImageCaptioner(
                    preload_models=False, 
                    use_gpu=current_use_gpu,
                    caption_model_name=model_name
                )
                
                # Sprawdź, czy model został poprawnie załadowany
                if not hasattr(captioner, 'caption_model') or captioner.caption_model is None:
                    raise ValueError("Model nie został poprawnie zainicjalizowany")
                
                start_time = time.time()
                
                # Generuj opis po polsku
                description = captioner.describe_image(image_path)
                
                # Sprawdź, czy opis nie jest pusty lub None
                if not description:
                    raise ValueError("Wygenerowany opis jest pusty")
                
                # Zapisz wynik
                results.append({
                    "path": image_path,
                    "description": description
                })
                
                successful_count += 1
                print(f"✓ Wygenerowano opis w {time.time() - start_time:.2f} sekund")
                print(f"Opis: {description[:100]}..." if len(description) > 100 else description)
                
                # Oznacz jako sukces i przerwij pętlę prób
                success = True
                break
                
            except Exception as e:
                print(f"✗ Błąd w próbie {attempt+1}: {str(e)}")
                # Sprawdź błąd dotyczący pliku stronicowania
                if "Plik stronicowania jest za mały" in str(e):
                    print("PORADA: Zwiększ rozmiar pliku stronicowania w ustawieniach Windows.")
                    print("Panel sterowania -> System -> Ustawienia zaawansowane -> Wydajność -> Ustawienia zaawansowane -> Pamięć wirtualna")
                
                # Jeśli to ostatnia próba, zapisz błąd
                if attempt == max_retries - 1:
                    results.append({
                        "path": image_path,
                        "description": f"Błąd po {max_retries} próbach: {str(e)}"
                    })
        
        # Jeśli nie udało się wygenerować opisu po wszystkich próbach
        if not success:
            print(f"Nie udało się wygenerować opisu dla {image_path} po {max_retries} próbach")
        
        # Zawsze zamykaj captioner po każdym obrazie
        if captioner is not None:
            try:
                captioner.cleanup()
            except:
                pass
            del captioner
            captioner = None
        
        # Zapisz częściowe wyniki po każdym obrazie
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Zapisano częściowe wyniki ({len(results)} obrazów)")
        except Exception as e:
            print(f"Ostrzeżenie: Nie można zapisać częściowych wyników: {e}")
            # Próba zapisania w katalogu tymczasowym
            temp_path = os.path.join(os.path.expanduser("~"), "opisy_zdjec_temp.json")
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Zapisano awaryjną kopię do: {temp_path}")
            except:
                print("Nie można zapisać nawet w katalogu tymczasowym")
        
        # Wymuś pełne czyszczenie pamięci po każdym obrazie
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        
        print("-" * 80)
    
    # Zapisz końcowe wyniki do JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wszystkie wyniki zapisano do {output_json_path}")
    except Exception as e:
        print(f"Błąd podczas zapisywania wyników końcowych: {e}")
        # Próba zapisania w katalogu tymczasowym
        temp_path = os.path.join(os.path.expanduser("~"), "opisy_zdjec_temp.json")
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Zapisano awaryjną kopię do: {temp_path}")
        except:
            print("Nie można zapisać nawet w katalogu tymczasowym")
    
    print(f"Statystyki: przetworzono {len(results)} obrazów, udanych opisów: {successful_count}")
    return output_json_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generowanie polskich opisów dla zdjęć")
    parser.add_argument("--folder", type=str, 
                      default=r"C:\Users\Maciej\Desktop\ZSD\pgverse\data\input1\images",
                      help="Ścieżka do folderu ze zdjęciami")
    parser.add_argument("--output", type=str, default=None,
                      help="Ścieżka do pliku wyjściowego JSON")
    parser.add_argument("--cpu", action="store_true",
                      help="Wymuś użycie CPU zamiast GPU")
    parser.add_argument("--wznow", type=int, default=None,
                      help="Wznów od podanego indeksu obrazu (licząc od 0)")
    parser.add_argument("--maly-model", action="store_true",
                      help="Użyj mniejszego modelu (blip2-flan-t5-xl) zamiast domyślnego")
    args = parser.parse_args()
    
    # Sprawdź czy folder ze zdjęciami istnieje
    if not os.path.exists(args.folder):
        print(f"Błąd: Folder {args.folder} nie istnieje.")
        sys.exit(1)
    
    # Dodatkowa informacja o zwiększeniu pliku stronicowania
    if not args.cpu:
        print("PORADA: Jeśli napotkasz błędy 'Plik stronicowania jest za mały', rozważ:")
        print("1. Użycie flagi --cpu, aby używać tylko CPU")
        print("2. Użycie flagi --maly-model, aby używać mniejszego modelu")
        print("3. Zwiększenie rozmiaru pliku stronicowania w Windows (zalecane min. 32 GB)")
    
    try:
        # Generuj opisy i zapisz do JSON
        result_path = create_image_descriptions_json(
            args.folder, 
            args.output,
            use_gpu=not args.cpu,
            resume_from=args.wznow
        )
        print(f"Proces zakończony. Plik JSON zapisany w: {result_path}")
    except KeyboardInterrupt:
        print("\nPraca przerwana przez użytkownika. Postęp został zapisany.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        print("Możesz wznowić pracę używając argumentu --wznow")