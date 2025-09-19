
"""
mozg_ocr.py — prosty runner: odpala skrypty jeden po drugim, bez przekazywania ścieżek.
"""

import sys
import subprocess


def run_script(script_path):
    """
    Funkcjonalność:
        Uruchamia wskazany skrypt Pythona w tym samym interpreterze.
        Jeśli skrypt zwróci błąd, zatrzymuje działanie programu.

    Args:
        script_path - ścieżka do pliku skryptu .py

    Returns:
        None
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



     # 4) Rozpoznawanie wzorów LaTeX i zapis JSON/TeX
    run_script("ocr_wzory_latex.py")


    run_script("ocr_ekstrakcja_z_tabel_img2_paddle.py")

    print("Wszystkie skrypty wykonane pomyślnie.")
