
"""
mozg_ocr.py — prosty runner: odpala skrypty jeden po drugim, bez przekazywania ścieżek.
"""

import sys
import subprocess


def run_script(script_path):
    """
    Uruchamia podany skrypt Pythona w tym samym interpreterze.
    Jeśli skrypt zakończy się błędem, przerywa działanie.
    """
    print(f"Uruchamiam: {script_path}...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Błąd: {script_path} zwrócił kod {result.returncode}. Przerywam.")
        sys.exit(result.returncode)
    print(f"Sukces: {script_path} zakończony pomyślnie.\n")


if __name__ == "__main__":
    # 1) Uruchomienie detekcji elementów
    run_script("detekcja_elementow.py")

    # 2) Uruchomienie OCR na wynikach detekcji
    run_script("wyodrebniony_tekst.py")

    print("Wszystkie skrypty wykonane pomyślnie.")
