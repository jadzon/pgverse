"""
mozg_ocr_updated.py — Updated version with command-line argument support
Usage: python mozg_ocr_updated.py --input_pdf path/to/file.pdf --subject subject_name --output_dir path/to/output
"""

import sys
import subprocess
import argparse
import os
import shutil
from pathlib import Path


def run_script(script_path, check_result=True):
    """
    Uruchamia podany skrypt Pythona w tym samym interpreterze.
    Jeśli skrypt zakończy się błędem, przerywa działanie.
    """
    print(f"Uruchamiam: {script_path}...")
    result = subprocess.run([sys.executable, script_path])
    if check_result and result.returncode != 0:
        print(f"Błąd: {script_path} zwrócił kod {result.returncode}. Przerywam.")
        sys.exit(result.returncode)
    print(f"Sukces: {script_path} zakończony pomyślnie.\n")


def prepare_input_files(input_pdf, subject, output_dir):
    """
    Prepare input files and directory structure
    """
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy input PDF to expected location (k1.pdf for now)
    target_pdf = "k1.pdf"
    if os.path.exists(target_pdf):
        os.remove(target_pdf)
    shutil.copy2(input_pdf, target_pdf)
    
    # Set environment variables for scripts to use
    os.environ["OCR_INPUT_PDF"] = str(input_pdf)
    os.environ["OCR_SUBJECT"] = subject
    os.environ["OCR_OUTPUT_DIR"] = str(output_dir)
    
    return target_pdf


def cleanup_temp_files(temp_files):
    """Clean up temporary files"""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Usunięto tymczasowy plik: {file_path}")
        except Exception as e:
            print(f"Nie można usunąć {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="OCR processing with subject classification")
    parser.add_argument("--input_pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--subject", required=True, help="Subject name for classification")
    parser.add_argument("--output_dir", help="Output directory (optional)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_pdf):
        print(f"Błąd: Plik {args.input_pdf} nie istnieje.")
        sys.exit(1)
    
    # Set default output directory if not provided
    if not args.output_dir:
        base_dir = Path(__file__).parent.parent.parent  # Go up to pgverse root
        args.output_dir = base_dir / "ocr" / "results" / args.subject / Path(args.input_pdf).stem
    
    print(f"Rozpoczynam OCR dla:")
    print(f"  Plik: {args.input_pdf}")
    print(f"  Przedmiot: {args.subject}")
    print(f"  Katalog wyników: {args.output_dir}")
    
    # Prepare input files
    temp_pdf = prepare_input_files(args.input_pdf, args.subject, args.output_dir)
    temp_files = [temp_pdf]
    
    try:
        # 1) Uruchomienie detekcji elementów
        run_script("detekcja_elementow.py")

        # 2) Uruchomienie OCR na wynikach detekcji
        run_script("wyodrebniony_tekst.py")

        # 3) Rozpoznawanie wzorów LaTeX i zapis JSON/TeX
        #run_script("ocr_wzory_latex.py")

        # 4) Ekstrakcja z tabel
        #run_script("ocr_ekstrakcja_z_tabel_img2_paddle.py")

        print("Wszystkie skrypty wykonane pomyślnie.")
        
        # Move results to final location
        ksiazki_dir = "ksiazki"
        if os.path.exists(ksiazki_dir):
            print(f"Przenoszę wyniki do {args.output_dir}")
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)
            shutil.move(ksiazki_dir, args.output_dir)
        
    except Exception as e:
        print(f"Błąd podczas przetwarzania: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files)


if __name__ == "__main__":
    main()
