import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import os
from tkinter import END
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import enviromental_variables as ev
from rag_utils import process_query  # Zakładam, że masz ten moduł z funkcją process_query

# Stałe konfiguracyjne
MODEL_NAME = ev.MODEL_NAME
MAX_TOKENS = ev.MAX_TOKENS
JSON_PATH = os.path.join(os.path.dirname(__file__), ev.JSON_PATH)

def load_model():
    # Sprawdzenie dostępności GPU dla 8-bitowej kwantyzacji
    if not torch.cuda.is_available():
        raise RuntimeError("Brak dostępnego GPU CUDA – 8-bitowa kwantyzacja wymaga CUDA.")
    device = torch.device("cuda")
    print(f"[INFO] Urządzenie: {device}")
    print("[INFO] Ładowanie tokenizer’a...")

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


class RagApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        # Inicjalizacja modelu
        self.tokenizer, self.model = load_model()
        # Ustawienia UI
        self.title("Bielik Chat")
        self.geometry("800x500")
        self.configure(bg='#001F3F')
        self.font_size = 12

        self.create_widgets()

        self.cohere_api_key =  ev.COHERE_API_KEY
        self.neo4j_uri = ev.NEO4J_URI
        self.neo4j_username = ev.NEO4J_USERNAME
        self.neo4j_password = ev.NEO4J_PASSWORD

        print("Model załadowany. Możesz pisać swoje prompty.")

    def create_widgets(self):
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
        zoom_in = tk.Button(input_frame, text='+', width=3, command=self.zoom_in, bg='#003366', fg='white')
        zoom_in.pack(side=tk.LEFT, padx=(0,5))
        zoom_out = tk.Button(input_frame, text='-', width=3, command=self.zoom_out, bg='#003366', fg='white')
        zoom_out.pack(side=tk.LEFT, padx=(0,10))

        self.user_input = tk.Entry(input_frame, font=("Arial", self.font_size), bg='#003366', fg='white', insertbackground='white')
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.user_input.bind("<Return>", lambda e: self.send_message())
        send_btn = tk.Button(input_frame, text='Wyślij', command=self.send_message, width=10, bg='#003366', fg='white')
        send_btn.pack(side=tk.RIGHT)

        self.chat_frame.bind("<Configure>", self._on_frame_configure)

    def zoom_in(self):
        self.font_size += 2
        self._update_font_size()

    def zoom_out(self):
        if self.font_size > 6:
            self.font_size -= 2
            self._update_font_size()

    def _update_font_size(self):
        self.user_input.config(font=("Arial", self.font_size))
        for child in self.chat_frame.winfo_children():
            if isinstance(child, tk.Label):
                child.config(font=("Arial", self.font_size))

    def send_message(self):
        prompt = self.user_input.get().strip()
        if not prompt:
            return
        self._add_message(f"Ty: {prompt}")
        self.user_input.delete(0, END)
        self._scroll_to_bottom()

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

        self._add_message(answer, references)
        self._scroll_to_bottom()

    def _add_message(self, content, references=None):
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

    def show_references(self, references):
        win = tk.Toplevel(self)
        win.title("Referencje i metryki")
        txt = ScrolledText(win, wrap=tk.WORD, font=("Arial", self.font_size))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for idx, ref in enumerate(references, 1):
            txt.insert(END, f"{idx}. {ref.get('reference')}\n")
            txt.insert(END, f"   Preision: {ref.get('precision'):.3f}, Recall: {ref.get('recall'):.3f}, F1: {ref.get('f1'):.3f}\n\n")
        txt.configure(state='disabled')
        btn_close = tk.Button(win, text='Zamknij', command=win.destroy, bg='#003366', fg='white')
        btn_close.pack(pady=5)

    def _on_frame_configure(self, event):
        self.chat_container.configure(scrollregion=self.chat_container.bbox('all'))
        # Aktualizacja wraplength dla wszystkich wiadomości przy zmianie rozmiaru
        for child in self.chat_frame.winfo_children():
            if isinstance(child, tk.Label):
                child.config(wraplength=self.chat_container.winfo_width()-20)

    def _scroll_to_bottom(self):
        self.chat_container.update_idletasks()
        self.chat_container.yview_moveto(1.0)


if __name__ == '__main__':
    app = RagApplication()
    app.mainloop()

