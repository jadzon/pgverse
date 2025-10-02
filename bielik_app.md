# Bielik Chat Application

## Opis

Aplikacja GUI do komunikacji z modelem językowym Bielik, wykorzystująca interfejs Tkinter i 8-bitową kwantyzację dla optymalizacji wydajności na GPU.

## Funkcjonalności

### Główne możliwości

- **Chat z modelem Bielik** - Interaktywna konwersacja z polskim modelem językowym
- **8-bitowa kwantyzacja** - Optymalizacja pamięci przez BitsAndBytesConfig
- **Skalowanie interfejsu** - Regulacja rozmiaru czcionki (+/-)
- **Przewijanie rozmowy** - Automatyczne scrollowanie do najnowszych wiadomości
- **Responsywny design** - Dostosowanie do zmiany rozmiaru okna

## Wymagania techniczne

### Biblioteki Python

```python
import tkinter as tk
from tkinter import END
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import enviromental_variables as ev
```

### Wymagania systemowe

- **GPU CUDA** - Obowiązkowe dla 8-bitowej kwantyzacji
- **PyTorch z obsługą CUDA**
- **Transformers library** z BitsAndBytesConfig
- **Model Bielik** - Polski model językowy

## Struktura aplikacji

### Funkcja load_model()

Odpowiada za ładowanie modelu i tokenizera z optymalizacją.

#### Konfiguracja kwantyzacji
```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None
)
```

#### Sprawdzanie GPU
- Weryfikuje dostępność CUDA przed ładowaniem
- Automatyczna mapa urządzeń (`device_map="auto"`)
- Wyrzuca RuntimeError przy braku GPU

### Klasa ChatApplication

Główna klasa dziedzicząca po `tk.Tk`, implementująca kompletny interfejs czatu.

#### Inicjalizacja
- Ładowanie modelu przy starcie aplikacji
- Ustawienie wymiarów okna 800x500
- Ciemny motyw kolorystyczny (#001F3F, #003366)
- Domyślny rozmiar czcionki 12px

#### Struktura interfejsu

**Główny obszar czatu:**
- Canvas z przewijaniem
- Frame z wiadomościami
- Scrollbar po prawej stronie

**Obszar wprowadzania:**
- Przyciski zoom (+/-)
- Pole tekstowe Entry
- Przycisk "Wyślij"
- Obsługa Enter do wysyłania

## Metody interfejsu

### Zarządzanie czcionką

- `zoom_in()` - Zwiększa rozmiar o 2px
- `zoom_out()` - Zmniejsza rozmiar (minimum 6px)
- `_update_font_size()` - Aktualizuje wszystkie elementy

### Obsługa wiadomości

- `send_message()` - Przetwarza zapytania użytkownika
- `_add_message(content)` - Dodaje wiadomość do czatu
- `_scroll_to_bottom()` - Automatyczne przewijanie

### Responsive design

- `_on_frame_configure()` - Obsługa zmiany rozmiaru
- Dynamiczne `wraplength` dla wiadomości
- Aktualizacja scrollregion

## Parametry generowania tekstu

### Konfiguracja modelu
```python
out = self.model.generate(
    **inputs,
    max_new_tokens=MAX_TOKENS,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    pad_token_id=self.tokenizer.eos_token_id
)
```

### Interpretacja parametrów

- **max_new_tokens** - Limit długości odpowiedzi (z ev.MAX_TOKENS)
- **temperature=0.7** - Zbalansowana kreatywność vs precyzja
- **top_p=0.95** - Nucleus sampling dla jakości odpowiedzi
- **do_sample=True** - Włączenie losowości w generowaniu

## Przetwarzanie odpowiedzi

### Ekstrakcja odpowiedzi
1. **Tokenizacja zapytania** - Konwersja na tensory PyTorch
2. **Generowanie** - Model tworzy odpowiedź z kontekstem
3. **Dekodowanie** - Konwersja z tokenów na tekst
4. **Czyszczenie** - Usuwanie oryginalnego promptu z odpowiedzi

### Optymalizacje
- `torch.no_grad()` - Wyłączenie gradientów dla inference
- Automatyczne przenoszenie na urządzenie modelu
- Skip special tokens przy dekodowaniu

## Kolorystyka i styling

### Paleta kolorów
- **Tło główne**: `#001F3F` (ciemny granat)
- **Elementy UI**: `#003366` (niebieski)
- **Tekst**: biały na ciemnym tle
- **Kursor**: biały w polu tekstowym

### Typografia
- **Font**: Arial w zmiennym rozmiarze
- **Justowanie**: left-aligned dla czytelności
- **Wrapping**: automatyczne łamanie długich wiadomości

## Obsługa błędów

### Sprawdzanie GPU
- Weryfikacja CUDA przed startem
- Komunikaty informacyjne o urządzeniu
- Graceful error handling

### Walidacja inputu
- Sprawdzanie pustych promptów
- Obsługa specjalnych znaków
- Bezpieczne czyszczenie pola tekstowego

## Uruchomienie aplikacji

```python
if __name__ == '__main__':
    app = ChatApplication()
    app.mainloop()
```

### Proces startowy
1. **Inicjalizacja** - Tworzenie instancji ChatApplication
2. **Ładowanie modelu** - Load_model() z konfiguracją 8-bit
3. **Tworzenie GUI** - create_widgets() buduje interfejs
4. **Mainloop** - Uruchomienie pętli zdarzeń Tkinter

## Optymalizacje wydajności

### Kwantyzacja 8-bitowa
- Redukcja zużycia pamięci GPU o ~50%
- Zachowanie jakości odpowiedzi
- Przyspieszenie inference na większych modelach

### Zarządzanie pamięcią
- Automatyczne garbage collection
- Optymalne device mapping
- Efektywne przetwarzanie tensorów

## Rozszerzenia

### Możliwe ulepszenia
- **Historia rozmów** - Zapisywanie sesji na dysk
- **Eksport czatu** - Możliwość zapisania jako TXT/PDF
- **Personalizacja** - Zmiana motywów kolorystycznych
- **Streaming** - Pokazywanie odpowiedzi w czasie rzeczywistym
- **Multi-turn** - Kontekst poprzednich wymian zdań