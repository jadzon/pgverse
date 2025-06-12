# Zintegrowana aplikacja RAG z wyszukiwaniem semantycznym obrazów
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

# Import your existing modules
import enviromental_variables as ev
from rag_utils import process_query

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Stałe konfiguracyjne
MODEL_NAME = ev.MODEL_NAME
MAX_TOKENS = ev.MAX_TOKENS
JSON_PATH = os.path.join(os.path.dirname(__file__), ev.JSON_PATH)

class SemanticSearchEngine:
    """Silnik wyszukiwania semantycznego obrazów"""
    
    def __init__(self):
        """Inicjalizacja silnika wyszukiwania semantycznego"""
        load_dotenv()
        
        # Konfiguracja Cohere
        self.cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
        
        # Konfiguracja Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        
        logger.info("SemanticSearchEngine zainicjalizowany pomyślnie")

    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generuje embedding dla zapytania użytkownika"""
        try:
            response = self.cohere_client.embed(
                texts=[query_text],
                model='embed-multilingual-v3.0',
                input_type='search_query',
                embedding_types=['float']
            )
            return response.embeddings.float[0]
        except Exception as e:
            logger.error(f"Błąd podczas generowania embeddingu zapytania: {e}")
            raise

    def find_similar_embeddings(self, query_embedding: List[float], threshold: float = 0.7, limit: int = 5) -> List[Dict[str, Any]]:
        """Znajduje podobne embeddingi w bazie Neo4j"""
        with self.neo4j_driver.session() as session:
            query = """
            CALL db.index.vector.queryNodes('embedding_vector', $limit, $query_embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.relative_path AS sciezka,
                   node.description AS opis,
                   node.content_type AS typ,
                   labels(node) AS etykiety,
                   score
            ORDER BY score DESC
            """
            
            try:
                result = session.run(query, {
                    'query_embedding': query_embedding,
                    'threshold': threshold,
                    'limit': limit
                })
                
                wyniki = []
                for record in result:
                    wyniki.append({
                        'sciezka': record['sciezka'],
                        'opis': record['opis'],
                        'typ': record['typ'],
                        'etykiety': record['etykiety'],
                        'podobienstwo': record['score']
                    })
                
                return wyniki
                
            except Exception as e:
                logger.error(f"Błąd podczas wyszukiwania w bazie: {e}")
                raise

    def search_best_image(self, query_text: str, threshold: float = 0.6) -> Optional[str]:
        """
        Znajduje najlepiej pasujący obraz do zapytania
        
        Args:
            query_text: Zapytanie tekstowe użytkownika
            threshold: Próg podobieństwa (domyślnie 0.6)
            
        Returns:
            Ścieżka do najlepiej pasującego obrazu lub None
        """
        logger.info(f"Wyszukuję obraz dla zapytania: '{query_text}'")
        
        try:
            # Generuj embedding dla zapytania
            query_embedding = self.generate_query_embedding(query_text)
            
            # Znajdź podobne embeddingi - preferuj obrazy (Figure)
            wyniki = self.find_similar_embeddings(query_embedding, threshold, limit=10)
            
            if not wyniki:
                logger.info("Nie znaleziono obrazów spełniających kryteria")
                return None
            
            # Preferuj wyniki typu Figure (obrazy)
            obrazy = [w for w in wyniki if w['typ'] == 'Figure']
            if obrazy:
                najlepszy = obrazy[0]  # Pierwszy (najlepiej pasujący)
                logger.info(f"Znaleziono obraz: {najlepszy['sciezka']} (podobieństwo: {najlepszy['podobienstwo']:.3f})")
                return najlepszy['sciezka']
            
            # Jeśli brak obrazów, weź najlepszy wynik dowolnego typu
            najlepszy = wyniki[0]
            logger.info(f"Znaleziono plik: {najlepszy['sciezka']} (podobieństwo: {najlepszy['podobienstwo']:.3f})")
            return najlepszy['sciezka']
            
        except Exception as e:
            logger.error(f"Błąd podczas wyszukiwania obrazu: {e}")
            return None

    def close(self):
        """Zamyka połączenia"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        logger.info("SemanticSearchEngine - połączenia zamknięte")


def load_model():
    """Ładuje model językowy"""
    if not torch.cuda.is_available():
        raise RuntimeError("Brak dostępnego GPU CUDA – 8-bitowa kwantyzacja wymaga CUDA.")
    device = torch.device("cuda")
    print(f"[INFO] Urządzenie: {device}")
    print("[INFO] Ładowanie tokenizer'a...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None
    )

    print("[INFO] Ładowanie modelu w trybie 8-bitowej kwantyzacji (bitsandbytes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return tokenizer, model


class IntegratedRagApplication(tk.Tk):
    """Zintegrowana aplikacja RAG z wyszukiwaniem semantycznym obrazów"""
    
    def __init__(self):
        super().__init__()
        
        # Inicjalizacja modelu językowego
        self.tokenizer, self.model = load_model()
        
        # Inicjalizacja silnika wyszukiwania obrazów
        self.search_engine = SemanticSearchEngine()
        
        # Ustawienia UI
        self.title("Bielik Chat z wyszukiwaniem obrazów")
        self.geometry("900x600")
        self.configure(bg='#001F3F')
        self.font_size = 12

        self.create_widgets()

        # Konfiguracja bazy danych
        self.cohere_api_key = ev.COHERE_API_KEY
        self.neo4j_uri = ev.NEO4J_URI
        self.neo4j_username = ev.NEO4J_USERNAME
        self.neo4j_password = ev.NEO4J_PASSWORD

        print("Model i silnik wyszukiwania załadowane. Możesz pisać swoje prompty.")

    def create_widgets(self):
        """Tworzy interfejs użytkownika"""
        # Główny frame
        main_frame = tk.Frame(self, bg='#001F3F')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas + Scrollbar
        self.chat_container = tk.Canvas(main_frame, bg='#001F3F', highlightthickness=0)
        self.chat_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0), pady=(10,0))
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=self.chat_container.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(10,60))
        self.chat_frame = tk.Frame(self.chat_container, bg='#001F3F')
        self.chat_container.create_window((0,0), window=self.chat_frame, anchor='nw')
        self.chat_container.configure(yscrollcommand=scrollbar.set)

        # Input frame
        input_frame = tk.Frame(self, bg='#001F3F')
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Przyciski zoom
        zoom_in = tk.Button(input_frame, text='+', width=3, command=self.zoom_in, bg='#003366', fg='white')
        zoom_in.pack(side=tk.LEFT, padx=(0,5))
        zoom_out = tk.Button(input_frame, text='-', width=3, command=self.zoom_out, bg='#003366', fg='white')
        zoom_out.pack(side=tk.LEFT, padx=(0,10))

        # Pole input
        self.user_input = tk.Entry(input_frame, font=("Arial", self.font_size), bg='#003366', fg='white', insertbackground='white')
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.user_input.bind("<Return>", lambda e: self.send_message())
        
        # Przycisk wyślij
        send_btn = tk.Button(input_frame, text='Wyślij', command=self.send_message, width=10, bg='#003366', fg='white')
        send_btn.pack(side=tk.RIGHT)

        self.chat_frame.bind("<Configure>", self._on_frame_configure)

    def zoom_in(self):
        """Zwiększa rozmiar czcionki"""
        self.font_size += 2
        self._update_font_size()

    def zoom_out(self):
        """Zmniejsza rozmiar czcionki"""
        if self.font_size > 6:
            self.font_size -= 2
            self._update_font_size()

    def _update_font_size(self):
        """Aktualizuje rozmiar czcionki"""
        self.user_input.config(font=("Arial", self.font_size))
        for child in self.chat_frame.winfo_children():
            if isinstance(child, tk.Label):
                child.config(font=("Arial", self.font_size))

    def send_message(self):
        """Wysyła wiadomość i przetwarza odpowiedź"""
        prompt = self.user_input.get().strip()
        if not prompt:
            return
            
        self._add_message(f"Ty: {prompt}")
        self.user_input.delete(0, END)
        self._scroll_to_bottom()

        try:
            # Przetwórz zapytanie przez RAG
            answer = process_query(prompt, 
                                   self.cohere_api_key, 
                                   self.neo4j_uri, 
                                   self.neo4j_username, 
                                   self.neo4j_password, 
                                   self.tokenizer, 
                                   self.model)
            
            # Pobierz metryki referencji z załadowanych danych JSON
            try:
                with open(JSON_PATH, 'r', encoding='utf-8') as f:
                    bert_scores = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                bert_scores = []

            references = []
            for entry in bert_scores:
                if entry.get('query') == prompt:
                    for score in entry.get('scores', []):
                        references.append({
                            'reference': score.get('reference', ''),
                            'precision': score.get('precision', 0.0),
                            'recall': score.get('recall', 0.0),
                            'f1': score.get('f1', 0.0)
                        })
                    break
            
            # Dodaj odpowiedź do czatu
            self._add_message(answer, references)
            
            # NOWA FUNKCJONALNOŚĆ: Wyszukaj i wyświetl pasujący obraz
            self._search_and_display_image(prompt)
            
        except Exception as e:
            error_msg = f"Błąd podczas przetwarzania zapytania: {e}"
            self._add_message(error_msg)
            logger.error(error_msg)

    def _search_and_display_image(self, query: str):
        """
        Wyszukuje i wyświetla obraz pasujący do zapytania
        
        Args:
            query: Zapytanie użytkownika
        """
        try:
            # Wyszukaj najlepiej pasujący obraz
            image_path = self.search_engine.search_best_image(query, threshold=0.5)
            
            if image_path:
                logger.info(f"Znaleziono obraz: {image_path}")
                
                # Sprawdź czy to względna czy bezwzględna ścieżka
                if not os.path.isabs(image_path):
                    # Jeśli względna, spróbuj różnych lokalizacji
                    possible_paths = [
                        image_path,  # Relative to current directory
                        os.path.join(os.path.dirname(__file__), image_path),  # Relative to script
                        os.path.join('data', image_path),  # In data folder
                        os.path.join('images', image_path),  # In images folder
                    ]
                    
                    found_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            found_path = path
                            break
                    
                    if found_path:
                        success = self._add_image_to_chat(found_path)
                        if success:
                            self._add_message(f"📷 Znaleziony obraz: {os.path.basename(image_path)}")
                        else:
                            self._add_message(f"⚠️ Nie można załadować obrazu: {image_path}")
                    else:
                        self._add_message(f"⚠️ Nie znaleziono pliku obrazu w żadnej z lokalizacji: {image_path}")
                else:
                    # Bezwzględna ścieżka
                    success = self._add_image_to_chat(image_path)
                    if success:
                        self._add_message(f"📷 Znaleziony obraz: {os.path.basename(image_path)}")
                    else:
                        self._add_message(f"⚠️ Nie można załadować obrazu: {image_path}")
            else:
                logger.info("Nie znaleziono pasującego obrazu")
                # Nie dodajemy komunikatu do czatu, żeby nie zaśmiecać interfejsu
                
        except Exception as e:
            logger.error(f"Błąd podczas wyszukiwania obrazu: {e}")
            self._add_message(f"⚠️ Błąd podczas wyszukiwania obrazu: {e}")

    def _add_message(self, content, references=None):
        """Dodaje wiadomość do czatu"""
        lbl = tk.Label(
            self.chat_frame,
            text=content,
            bg='#003366',
            fg='white',
            font=("Arial", self.font_size),
            wraplength=self.winfo_width()-40,
            justify='left'
        )
        lbl.pack(fill=tk.X, pady=2, padx=5)
        
        if references:
            lbl.references = references
            lbl.bind('<Button-1>', lambda e: self.show_references(e.widget.references))
        
        self._scroll_to_bottom()

    def _add_image_to_chat(self, image_path_or_url):
        """Dodaje obraz do czatu"""
        try:
            # Sprawdź czy to względna ścieżka i zbuduj pełną ścieżkę
            if not os.path.isabs(image_path_or_url):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                full_image_path = os.path.join(script_dir, image_path_or_url)
            else:
                full_image_path = image_path_or_url
            
            # Sprawdź czy plik istnieje
            if not os.path.exists(full_image_path):
                raise FileNotFoundError(f"Plik nie istnieje: {full_image_path}")
            
            image = Image.open(full_image_path)
            
            # Przeskaluj obraz do odpowiedniego rozmiaru
            max_width = 500
            max_height = 400
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Konwertuj do formatu Tkinter
            photo = ImageTk.PhotoImage(image)
            
            # Utwórz label z obrazem
            img_label = tk.Label(
                self.chat_frame,
                image=photo,
                bg='#001F3F'
            )
            img_label.image = photo  # Zachowaj referencję
            img_label.pack(pady=5, padx=5)
            
            self._scroll_to_bottom()
            return True
            
        except Exception as e:
            logger.error(f"Nie można załadować obrazu: {e}")
            return False

    def show_references(self, references):
        """Pokazuje okno z referencjami"""
        win = tk.Toplevel(self)
        win.title("Referencje i metryki")
        win.geometry("600x400")
        
        txt = ScrolledText(win, wrap=tk.WORD, font=("Arial", self.font_size))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for idx, ref in enumerate(references, 1):
            txt.insert(END, f"{idx}. {ref.get('reference')}\n")
            txt.insert(END, f"   Precision: {ref.get('precision'):.3f}, Recall: {ref.get('recall'):.3f}, F1: {ref.get('f1'):.3f}\n\n")
        
        txt.configure(state='disabled')
        
        btn_close = tk.Button(win, text='Zamknij', command=win.destroy, bg='#003366', fg='white')
        btn_close.pack(pady=5)

    def _on_frame_configure(self, event):
        """Callback dla zmiany rozmiaru ramki"""
        self.chat_container.configure(scrollregion=self.chat_container.bbox('all'))
        # Aktualizacja wraplength dla wszystkich wiadomości przy zmianie rozmiaru
        for child in self.chat_frame.winfo_children():
            if isinstance(child, tk.Label):
                child.config(wraplength=self.chat_container.winfo_width()-20)

    def _scroll_to_bottom(self):
        """Przewija czat na dół"""
        self.chat_container.update_idletasks()
        self.chat_container.yview_moveto(1.0)

    def destroy(self):
        """Zamyka aplikację i czyści zasoby"""
        try:
            if hasattr(self, 'search_engine'):
                self.search_engine.close()
        except Exception as e:
            logger.error(f"Błąd podczas zamykania silnika wyszukiwania: {e}")
        finally:
            super().destroy()


if __name__ == '__main__':
    app = IntegratedRagApplication()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("\nZamykanie aplikacji...")
    finally:
        app.destroy()