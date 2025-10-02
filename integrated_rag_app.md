# Zintegrowana aplikacja RAG z wyszukiwaniem semantycznym obrazów

## Opis

Aplikacja GUI z interfejsem Tkinter, która łączy model językowy z wyszukiwaniem semantycznym obrazów. Wykorzystuje bazę danych Neo4j i API Cohere do generowania embeddingów i wyszukiwania semantycznego.

## Funkcjonalności

### Główne komponenty

- **SemanticSearchEngine** - Silnik wyszukiwania semantycznego obrazów
- **IntegratedRagApplication** - Główna aplikacja z interfejsem GUI
- **RAG Integration** - Integracja z systemem Retrieval-Augmented Generation

### Funkcje aplikacji

- Czat z modelem językowym (Bielik)
- Wyszukiwanie semantyczne obrazów na podstawie zapytań tekstowych
- Wyświetlanie znalezionych obrazów w czacie
- Skalowanie czcionki w interfejsie
- Wyświetlanie metryk referencji (BERT scores)

## Wymagania techniczne

### Biblioteki Python

```python
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import os
from tkinter import END
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image, ImageTk
from tkinter import filedialog
import requests
from io import BytesIO
import logging
from typing import List, Dict, Any, Optional
import cohere
from neo4j import GraphDatabase
from dotenv import load_dotenv
```

### Wymagania systemowe

- **GPU CUDA** - wymagane do 8-bitowej kwantyzacji modelu
- **Neo4j** - baza danych grafowa z indeksem wektorowym
- **Cohere API** - do generowania embeddingów

## Klasy i metody

### SemanticSearchEngine

Główny silnik wyszukiwania semantycznego.

#### Metody

- `__init__()` - Inicjalizacja z konfiguracją Cohere i Neo4j
- `generate_query_embedding(query_text: str)` - Generuje embedding dla zapytania
- `find_similar_embeddings(query_embedding, threshold, limit)` - Wyszukuje podobne embeddingi w bazie
- `search_best_image(query_text: str, threshold: float = 0.8)` - Znajduje najlepiej pasujący obraz
- `close()` - Zamyka połączenia z bazą danych

### IntegratedRagApplication

Główna klasa aplikacji dziedzicząca po `tk.Tk`.

#### Funkcjonalności interfejsu

- Chat z przewijaniem
- Pole wprowadzania tekstu
- Przyciski zoom (+/-)
- Wyświetlanie obrazów
- Okna z referencjami i metrykami

#### Główne metody

- `create_widgets()` - Tworzy elementy interfejsu
- `send_message()` - Przetwarza wiadomości użytkownika
- `_search_and_display_image(query)` - Wyszukuje i wyświetla obrazy
- `_add_message(content, references)` - Dodaje wiadomość do czatu
- `_add_image_to_chat(image_path)` - Dodaje obraz do czatu
- `show_references(references)` - Wyświetla okno z referencjami

## Konfiguracja

### Zmienne środowiskowe

Aplikacja wymaga konfiguracji następujących zmiennych:

- `COHERE_API_KEY` - Klucz API do Cohere
- `NEO4J_URI` - URI bazy danych Neo4j
- `NEO4J_USER` - Nazwa użytkownika Neo4j
- `NEO4J_PASSWORD` - Hasło do Neo4j

### Model językowy

- Domyślnie używany jest model z `ev.MODEL_NAME`
- 8-bitowa kwantyzacja przez BitsAndBytesConfig
- Automatyczna mapa urządzeń dla GPU

## Funkcje wyszukiwania obrazów

### Algorytm wyszukiwania

1. **Generowanie embeddingu** - Zapytanie użytkownika jest konwertowane na embedding przez Cohere API
2. **Wyszukiwanie w bazie** - Embedding jest porównywany z embeddingami w Neo4j
3. **Filtrowanie wyników** - Preferowane są obrazy (typ 'image')
4. **Wyświetlanie** - Najlepiej pasujący obraz jest wyświetlany w czacie

### Obsługa ścieżek

- Automatyczne usuwanie prefiksów `/pgverse/`
- Obsługa ścieżek względnych i bezwzględnych
- Sprawdzanie wielu możliwych lokalizacji plików

## Interfejs użytkownika

### Layout

- **Główny obszar czatu** - ScrolledText z Canvas i Scrollbar
- **Obszar wpisywania** - Entry field z przyciskami
- **Kontrolki zoom** - Przyciski +/- do skalowania czcionki

### Kolory i styling

- Tło aplikacji: `#001F3F` (ciemny niebieski)
- Elementy UI: `#003366` (niebieski)
- Tekst: biały
- Font: Arial z regulowanym rozmiarem

## Obsługa błędów

- Kompleksowe logowanie przez moduł `logging`
- Try-catch bloki dla operacji krytycznych
- Komunikaty błędów wyświetlane w czacie
- Graceful shutdown z zamykaniem połączeń

## Uruchomienie

```python
if __name__ == '__main__':
    app = IntegratedRagApplication()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("\nZamykanie aplikacji...")
    finally:
        app.destroy()
```

## Dodatkowe funkcje

### Wyświetlanie obrazów

- Automatyczne skalowanie do rozmiaru 800x600
- Kliknięcie na obraz otwiera pełny rozmiar w nowym oknie
- Obsługa różnych formatów obrazów przez PIL

### Metryki i referencje

- Ładowanie metryk BERT z pliku JSON
- Wyświetlanie precision, recall i F1 score
- Klikalne etykiety z referencjami

### Zarządzanie zasobami

- Automatyczne zamykanie połączeń z bazą danych
- Proper cleanup przy zamykaniu aplikacji
- Obsługa KeyboardInterrupt