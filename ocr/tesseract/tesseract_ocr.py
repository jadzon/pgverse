import os
import pytesseract
from pdf2image import convert_from_path

# Ścieżka do pliku PDF w folderze projektu
pdf_file = "D:\\nauka\\baza\ksiazka 3\\k3pdf.pdf"

# Ścieżka do Tesseracta (jeśli używasz Windowsa, podaj pełną ścieżkę)
pytesseract.pytesseract.tesseract_cmd = 'D:\\nauka\\tesseract z instalatora\\tesseract.exe'

# Konwersja PDF na listę obrazów
images = convert_from_path(pdf_file)

# Plik wyjściowy
output_txt = "wynik7.txt"

with open(output_txt, "w", encoding="utf-8") as f:
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="pol",config="--psm 4")  # Możesz zmienić 'pol' na inny język
        f.write(f"=== Strona {i+1} ===\n{text}\n\n")
        
        # Informacja w terminalu po przetworzeniu każdej strony
        print(f"Przetworzono stronę {i+1} / {len(images)}")

print(f"Wynik zapisany do {output_txt}")
