import re
import os

def clean_ocr_text(text):
    """
    Czyści tekst pochodzący z OCR z typowych błędów i formatowań.
    """
    # Usunięcie komentarzy z pliku (jeśli są znaczniki filepath)
    text = re.sub(r'//\s*filepath:.*?\n', '', text)
    
    # Usunięcie numerów stron
    text = re.sub(r'=== Strona \d+ ===', '', text)
    
    # Usunięcie numerów linii i kolumn w stylu podpisów (np. "Rys. 4.18.")
    text = re.sub(r'Rys\.\s+\d+\.\d+\..*?(?=\n)', '', text)
    
    # Usunięcie odniesień do tabel
    text = re.sub(r'Tabela \d+\..*?(?=\n)', '', text)
    
    # Usunięcie numeracji stron na dole
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Usunięcie dzielenia wyrazów z pomocą myślników na końcu linii
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Złączenie wyrazów przerwanych między liniami
    text = re.sub(r'(\w+)\n(\w+)', r'\1 \2', text)
    
    # Usunięcie wielu spacji pod rząd
    text = re.sub(r' +', ' ', text)
    
    # Usunięcie dziwnych sekwencji literatury o postaci [1], [2] itp.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Usunięcie numeracji odnośników z podpisów rysunków (np. "1- tarcza, 2 - źródło światła")
    text = re.sub(r'\d+[\s-]+', '', text)
    
    # Usunięcie wielokrotnych pustych linii
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Poprawa dziwnych znaków interpunkcyjnych
    text = text.replace(' - ', ' — ')
    text = text.replace(' -', ' —')
    text = text.replace('- ', '— ')
    
    # Usunięcie specjalnych znaków formatujących
    text = re.sub(r'[""„«»]', '"', text)
    
    # Usunięcie numerów sekcji typu "4.5."
    text = re.sub(r'\d+\.\d+\.', '', text)
    
    # Usunięcie liter i liczb używanych do oznaczania wyliczenia
    text = re.sub(r'\n[a-z]\)\s*', '\n', text)
    text = re.sub(r'\n\d+\)\s*', '\n', text)
    
    # Usunięcie symboli matematycznych które mogły być źle zinterpretowane
    text = re.sub(r'\+\d+\^', '', text)
    
    # Połączenie fragmentów które zostały podzielone przez przypadkowe znaki nowej linii
    text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)
    
    # Naprawienie powszechnych błędów OCR
    text = text.replace('l-', '1-')
    text = text.replace('Il -', 'II -')
    text = text.replace('tp.', 'itp.')
    
    # Zamiana na prawidłowe jednostki
    text = re.sub(r'(\d+)[\+\-](\d+)', r'\1e\2', text)  # notacja wykładnicza
    
    return text

def split_into_coherent_sections(text):
    """
    Dzieli tekst na spójne sekcje tematyczne.
    """
    # Podział na sekcje po wykryciu nagłówków
    sections = []
    current_section = []
    
    # Wzór do wykrywania nagłówków (Wielkie litery na początku linii)
    header_pattern = r'\n([A-ZŚĄĘŹŻŃŁÓĆ][A-ZŚĄĘŹŻŃŁÓĆ\s]+)(?:\n|$)'
    
    # Dzielimy tekst na sekcje zgodnie z nagłówkami
    matches = re.finditer(header_pattern, text)
    last_end = 0
    
    for match in matches:
        if last_end < match.start():
            section_text = text[last_end:match.start()].strip()
            if section_text:
                sections.append(section_text)
        
        # Zapisz nagłówek jako osobną sekcję
        header = match.group(1).strip()
        sections.append(header)
        last_end = match.end()
    
    # Dodaj ostatnią sekcję
    if last_end < len(text):
        section_text = text[last_end:].strip()
        if section_text:
            sections.append(section_text)
    
    return sections

def remove_figure_captions_and_notes(sections):
    """
    Usuwa podpisy pod rysunkami i notatki, które nie zawierają wartościowych informacji.
    """
    cleaned_sections = []
    for section in sections:
        # Usuń podpisy pod rysunkami (zwykle zaczynają się od "Rys.")
        if not section.startswith("Rys.") and len(section.split()) > 5:
            # Usuń odnośniki do numerów, które nie stanowią wartościowych informacji
            if not re.match(r'^\d+[\.,]?\s*$', section):
                cleaned_sections.append(section)
    
    return cleaned_sections

def repair_paragraphs(text):
    """
    Naprawia paragrafy, które zostały przerwane przez przypadkowe podziały.
    """
    # Łączymy linie które nie kończą się kropką lub innym znakiem końca zdania
    lines = text.split('\n')
    result = []
    current_paragraph = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                result.append(current_paragraph)
                current_paragraph = ""
            continue
            
        # Jeśli linia nie kończy się znakiem końca zdania, połącz ją z następną
        if current_paragraph and not re.search(r'[.!?:]$', current_paragraph):
            current_paragraph += " " + line
        else:
            if current_paragraph:
                result.append(current_paragraph)
            current_paragraph = line
    
    if current_paragraph:
        result.append(current_paragraph)
        
    return '\n\n'.join(result)

def save_processed_text(text, output_file):
    """
    Zapisuje przetworzony tekst do pliku.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

def preprocess_ocr_file(input_file, output_file):
    """
    Główna funkcja do preprocessingu pliku OCR.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Etap 1: Czyszczenie tekstu z błędów OCR
        cleaned_text = clean_ocr_text(text)
        
        # Etap 2: Naprawienie paragrafów
        repaired_text = repair_paragraphs(cleaned_text)
        
        # Etap 3: Podział na sekcje tematyczne
        sections = split_into_coherent_sections(repaired_text)
        
        # Etap 4: Usunięcie podpisów pod rysunkami i nieistotnych sekcji
        cleaned_sections = remove_figure_captions_and_notes(sections)
        
        # Etap 5: Połączenie oczyszczonych sekcji z powrotem w tekst
        final_text = '\n\n'.join(cleaned_sections)
        
        # Ostateczne czyszczenie
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)  # Usuń wiele pustych linii
        final_text = re.sub(r' {2,}', ' ', final_text)  # Usuń wiele spacji
        
        # Zapisz przetworzony tekst
        save_processed_text(final_text, output_file)
        
        print(f"Preprocessing zakończony. Zapisano przetworzony tekst do {output_file}")
        
    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku: {e}")

if __name__ == "__main__":
    input_file = 'input.txt'
    output_file = 'processed_input.txt'
    
    if not os.path.isfile(input_file):
        print(f"Plik {input_file} nie istnieje.")
    else:
        preprocess_ocr_file(input_file, output_file)