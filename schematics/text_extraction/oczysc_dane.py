import os
import shutil
import sys
import time
import stat
import subprocess
import platform

def zmien_atrybuty_i_usun(sciezka):
    """
    Funkcja pomocnicza do zmiany atrybutów pliku/folderu i próby usunięcia.
    Działa głównie na systemach Windows w przypadku błędów dostępu.
    
    Args:
        sciezka: Ścieżka do elementu do usunięcia
        
    Returns:
        bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
    """
    try:
        # Próba zmiany atrybutów pliku (tylko dla Windows)
        if platform.system() == 'Windows':
            # Zmiana atrybutów na normalne
            os.chmod(sciezka, stat.S_IWRITE)
            # Dodatkowa próba przez attrib (tylko Windows)
            subprocess.run(['attrib', '-R', '-S', '-H', sciezka], check=False, shell=True)
            time.sleep(0.5)  # Dajemy czas na aplikację zmian
            
        # Próba usunięcia
        if os.path.isdir(sciezka):
            shutil.rmtree(sciezka)
        else:
            os.remove(sciezka)
        return True
    except Exception as e:
        print(f"    Alternatywna metoda również się nie powiodła: {e}")
        return False

def oczysc_folder(sciezka):
    """
    Czyści zawartość podanego folderu, zachowując sam folder.
    
    Args:
        sciezka: Ścieżka do folderu do wyczyszczenia
    
    Returns:
        bool: True jeśli udało się wyczyścić całkowicie, 
              False jeśli wystąpiły błędy lub folder nie istnieje
    """
    if not os.path.exists(sciezka):
        print(f"Folder {sciezka} nie istnieje, pomijam.")
        return False
    
    if not os.path.isdir(sciezka):
        print(f"{sciezka} nie jest folderem, pomijam.")
        return False
    
    bledy = 0
    sukcesy = 0
    
    # Próbujemy utworzyć folder (jeśli nie istnieje)
    try:
        os.makedirs(sciezka, exist_ok=True)
    except Exception as e:
        print(f"Nie mogę utworzyć/sprawdzić folderu {sciezka}: {e}")
    
    # Usuwamy zawartość folderu (pliki i podfoldery)
    for element in os.listdir(sciezka):
        pelna_sciezka = os.path.join(sciezka, element)
        
        try:
            if os.path.isdir(pelna_sciezka):
                shutil.rmtree(pelna_sciezka)
                print(f"  - Usunięto podfolder: {element}")
            else:
                os.remove(pelna_sciezka)
                print(f"  - Usunięto plik: {element}")
            sukcesy += 1
        except PermissionError as e:
            print(f"  ! Błąd uprawnień przy usuwaniu {pelna_sciezka}: {e}")
            print("    Próbuję alternatywną metodę...")
            if zmien_atrybuty_i_usun(pelna_sciezka):
                print(f"    Udało się usunąć {element} alternatywną metodą.")
                sukcesy += 1
            else:
                print(f"    Nie udało się usunąć {element}. Sprawdź czy plik nie jest używany przez inny program.")
                bledy += 1
        except Exception as e:
            print(f"  ! Błąd podczas usuwania {pelna_sciezka}: {e}")
            bledy += 1
    
    if bledy == 0:
        print(f"Wyczyszczono folder: {sciezka} (usunięto {sukcesy} elementów)")
        return True
    else:
        print(f"Częściowo wyczyszczono folder: {sciezka} (usunięto {sukcesy} elementów, {bledy} nie udało się usunąć)")
        return False

def main():
    print("=== Czyszczenie tymczasowych danych programu ===")
    
    # Lista folderów do wyczyszczenia
    foldery_do_wyczyszczenia = [
        "Dataset/Przetworzone",
        "text_marked",
        "visualization_output",
        "Wyniki_OCR"
    ]
    
    # Dodatkowe instrukcje dla Windows
    if platform.system() == 'Windows':
        print("\nUWAGA! W systemie Windows mogą wystąpić problemy z uprawnieniami.")
        print("Zalecane jest zamknięcie wszystkich programów, które mogą używać tych plików")
        print("(np. przeglądarki obrazów, edytory tekstu, itp.).")
    
    # Pytamy o potwierdzenie
    print("\nProgram usunie zawartość następujących folderów:")
    for folder in foldery_do_wyczyszczenia:
        print(f" - {folder}")
    
    potwierdzenie = input("\nCzy chcesz kontynuować? (t/n): ")
    if potwierdzenie.lower() not in ['t', 'tak', 'y', 'yes']:
        print("Operacja anulowana.")
        sys.exit(0)
    
    # Dla Windows, dodatkowe ostrzeżenie o prawach administratora
    if platform.system() == 'Windows':
        print("\nJeśli pojawią się błędy uprawnień, spróbuj uruchomić skrypt jako administrator.")
        time.sleep(1)  # Dajemy czas na przeczytanie
    
    # Czyszczenie folderów
    pelny_sukces = 0
    czesciowy_sukces = 0
    for folder in foldery_do_wyczyszczenia:
        print(f"\nCzyszczę folder: {folder}")
        if oczysc_folder(folder):
            pelny_sukces += 1
        else:
            czesciowy_sukces += 1
    
    # Podsumowanie
    print("\n=== Podsumowanie czyszczenia ===")
    print(f"Całkowicie wyczyszczono: {pelny_sukces} folderów")
    print(f"Częściowo wyczyszczono: {czesciowy_sukces} folderów")
    print(f"Razem przetworzono: {len(foldery_do_wyczyszczenia)} folderów")
    
    if czesciowy_sukces > 0:
        print("\nWskazówki w przypadku błędów:")
        print("1. Upewnij się, że żaden program nie ma otwartych plików w tych folderach")
        print("2. Zamknij wszystkie okna Eksploratora Windows pokazujące te foldery")
        print("3. Uruchom ten skrypt jako administrator")
        print("4. W ostateczności, spróbuj ponownie po restarcie komputera")

if __name__ == "__main__":
    main() 