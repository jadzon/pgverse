import os
import PyPDF2

def detect_document_type(file_path: str, text_threshold: int = 50) -> str:
    """
      - Jeśli uda się wyekstrahować dużo tekstu, dokument uznaje za wektorowy.
      - Jeśli wyekstrahowany tekst jest krótki lub pusty, dokument uznaje za skanowany.

    Parametry:
      file_path: Ścieżka do pliku PDF.
      text_threshold: Minimalna liczba znaków, aby uznać dokument za tekst wektorowy.

    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext != ".pdf":
        return "nieobsługiwany"

    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            extracted_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text
        # Jeśli wyekstrahowany tekst ma więcej znaków niż threshold, uznajemy za wektorowy
        if len(extracted_text.strip()) >= text_threshold:
            return "tekst wektorowy"
        else:
            return "skan"
    except Exception as e:
        print(f"Błąd podczas odczytu PDF: {e}")
        return "nieobsługiwany"

def extract_text_from_pdf(file_path: str) -> str:
    """
    Wyekstrahuje i zwraca tekst z dokumentu PDF.
    """
    extracted_text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
    except Exception as e:
        print(f"Błąd podczas ekstrakcji tekstu: {e}")
    return extracted_text

if __name__ == "__main__":
    file_path = input("Podaj ścieżkę do pliku PDF: ")
    doc_type = detect_document_type(file_path)
    if doc_type == "tekst wektorowy":
        print("Dokument jest wektorowy (z osadzonym tekstem).")
        text = extract_text_from_pdf(file_path)
        print("\nWyekstrahowany tekst:")
        print(text)
    elif doc_type == "skan":
        print("Dokument wygląda na skan (głównie obraz).")
    else:
        print("Nieobsługiwany typ pliku.")
