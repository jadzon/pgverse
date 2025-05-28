import re
import os

def clean_ocr_text(text):
    """
    Czyści tekst pochodzący z OCR z typowych błędów i formatowań.
    """
    # Usunięcie komentarzy z pliku (jeśli są znaczniki filepath)
    text = re.sub(r'//\s*filepath:.*?\n', '', text)
    
    # Usunięcie nagłówków stron
    text = re.sub(r'=== Strona \d+ ===', '', text)
    
    # Usunięcie numerów stron w postaci liczby na samodzielnej linii
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Usunięcie stopek stron (typowe dla czasopism naukowych)
    text = re.sub(r'\n.*?Zeszyty Problemowe — Maszyny Elektryczne Nr \d+/\d+.*?\n', '\n', text)
    
    # Lepsze rozpoznawanie i usuwanie podpisów rysunków (jednolinijkowe i wielolinijkowe)
    text = re.sub(r'Rys\.?\s+\d+\.?\d*\.?.*?(?=\n\n|\n[A-Z])', '', text)
    
    # Usunięcie odniesień do tabel
    text = re.sub(r'Tabela \d+\..*?(?=\n)', '', text)
    
    # Połączenie wyrazów przerwanych myślnikiem na końcu linii
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Usunięcie dziwnych oznaczeń elementów: Il -, l-
    text = re.sub(r'\s+[Ill]-\s+', ' 1- ', text)
    text = re.sub(r'\s+[Ill]\s*-\s*', ' 1- ', text)
    
    # Poprawa oznaczeń numerycznych elementów (1-, 2-, itp.)
    text = re.sub(r'([0-9])\s*-\s+', r'\1- ', text)
    
    # Łączenie paragrafów przerwanych w środku zdania (gdy linia nie kończy się kropką)
    lines = text.split('\n')
    result = []
    current_paragraph = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                result.append(current_paragraph)
                current_paragraph = ""
            result.append("")
            continue
            
        # Jeśli poprzednia linia nie kończy się znakiem końca zdania i nie jest nagłówkiem
        if current_paragraph and not re.search(r'[.!?:]\s*$', current_paragraph) and not re.match(r'^[0-9]+\.[0-9]+\.', line):
            current_paragraph += " " + line
        else:
            if current_paragraph:
                result.append(current_paragraph)
            current_paragraph = line
    
    if current_paragraph:
        result.append(current_paragraph)
    
    text = '\n'.join(result)
    
    # Usunięcie wielokrotnych pustych linii
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Poprawa dziwnych znaków interpunkcyjnych
    text = text.replace(' - ', ' — ')
    text = text.replace(' -', ' —')
    text = text.replace('- ', '— ')
    
    # Usunięcie specjalnych znaków formatujących
    text = re.sub(r'[""„«»]', '"', text)
    
    # Lepsze rozpoznawanie numerów sekcji
    text = re.sub(r'^\d+\.\d+\.(\s+[A-ZŚĄĘŹŻŃŁÓĆ])', r'\1', text, flags=re.MULTILINE)
    
    # Lepsze rozpoznawanie list wypunktowanych i numerowanych
    text = re.sub(r'\n\s*([a-z]|[0-9]+)[\)\.][\s\-]+', '\n• ', text)
    
    # Poprawa typowych błędów OCR
    text = text.replace('l-', '1-')
    text = text.replace('Il -', 'II -')
    text = text.replace('tp.', 'itp.')
    
    # Poprawa notacji naukowej i jednostek
    text = re.sub(r'(\d+)[\+\-](\d+)', r'\1e\2', text)  # notacja wykładnicza
    text = re.sub(r'([0-9])[\s\-]*\+[\s\-]*([0-9])', r'\1+\2', text)
    
    # Usunięcie wielu spacji pod rząd
    text = re.sub(r' +', ' ', text)
    
    return text

def split_into_coherent_sections(text):
    """
    Dzieli tekst na spójne sekcje tematyczne.
    Rozpoznaje nagłówki numerowane i nienumerowane.
    """
    # Uzupełniony wzorzec na nagłówki (uwzględniający numerowane sekcje)
    header_patterns = [
        r'\n(\d+\.\d+\.?\s+[A-ZŚĄĘŹŻŃŁÓĆ][A-ZĄĘŚŹŻŃŁÓĆa-ząęśźżńłóć\s]+)(?:\n|$)',  # Nagłówki numerowane
        r'\n([A-ZŚĄĘŹŻŃŁÓĆ][A-ZŚĄĘŹŻŃŁÓĆ\s]+)(?:\n|$)'  # Nagłówki WIELKIMI LITERAMI
    ]
    
    sections = []
    last_end = 0
    
    # Zbierz wszystkie pozycje nagłówków
    headers = []
    for pattern in header_patterns:
        for match in re.finditer(pattern, text):
            headers.append((match.start(), match.end(), match.group(1).strip()))
    
    # Sortuj nagłówki według pozycji
    headers.sort()
    
    for start, end, header in headers:
        if last_end < start:
            section_text = text[last_end:start].strip()
            if section_text:
                sections.append(section_text)
        
        # Zapisz nagłówek jako osobną sekcję
        sections.append(header)
        last_end = end
    
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
        # Rozszerzone wzorce dla podpisów rysunków
        if not (re.match(r'^(Rys|Tab|Tabela|Rysunek)\.\s*', section) or
                len(section.split()) <= 5 or
                re.match(r'^\d+[\.,]?\s*$', section) or
                re.match(r'^[0-9\-]+\s*[a-z]', section)):  # Elementy listy jak "1- tarcza"
            cleaned_sections.append(section)
    
    return cleaned_sections

def repair_tables(text):
    """
    Naprawia i formatuje tabele w tekście.
    """
    # Identyfikacja obszarów tabel
    table_pattern = r'Tabela \d+\..*?[\n\r].*?[\n\r]'
    
    def format_table_content(match):
        table_text = match.group(0)
        # Tu można dodać logikę formatowania tabel
        return "\n\nTABELA:\n" + table_text + "\n\n"
    
    return re.sub(table_pattern, format_table_content, text)

def preprocess_ocr_file(input_file, output_file):
    """
    Główna funkcja do preprocessingu pliku OCR.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Etap 1: Czyszczenie tekstu z błędów OCR
        cleaned_text = clean_ocr_text(text)
        
        # Etap 2: Formatowanie tabel
        cleaned_text = repair_tables(cleaned_text)
        
        # Etap 3: Naprawienie paragrafów (już zawarte w clean_ocr_text)
        
        # Etap 4: Podział na sekcje tematyczne
        sections = split_into_coherent_sections(cleaned_text)
        
        # Etap 5: Usunięcie podpisów pod rysunkami i nieistotnych sekcji
        cleaned_sections = remove_figure_captions_and_notes(sections)
        
        # Etap 6: Połączenie oczyszczonych sekcji z powrotem w tekst
        final_text = '\n\n'.join(cleaned_sections)
        
        # Ostateczne czyszczenie
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)  # Usuń wiele pustych linii
        final_text = re.sub(r' {2,}', ' ', final_text)  # Usuń wiele spacji
        
        # Zapisz przetworzony tekst
        save_processed_text(final_text, output_file)
        
        print(f"Preprocessing zakończony. Zapisano przetworzony tekst do {output_file}")
        
    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku: {e}")

def save_processed_text(text, output_file):
    """
    Zapisuje przetworzony tekst do pliku.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    input_dir = "input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Utworzono katalog {input_dir}")
        
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    if not input_files:
        print(f"Brak plików w katalogu {input_dir}. Umieść pliki do przetworzenia w tym katalogu.")
    else:
        for input_file in input_files:
            file_path = os.path.join(input_dir, input_file)
            output_file = f"processed_{input_file}"
            print(f"Przetwarzanie pliku: {input_file}...")
            preprocess_ocr_file(file_path, output_file)