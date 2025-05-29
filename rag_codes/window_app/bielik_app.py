import tkinter as tk
from tkinter import END
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Stałe konfiguracyjne
MODEL_NAME = "speakleash/Bielik-11B-v2.3-Instruct"
MAX_TOKENS = 300


def load_model():
    """
    Ładuje model i tokenizer z konfiguracją 4-bitową.
    """
    print("Ładowanie modelu i tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto"
    )

    return tokenizer, model

class ChatApplication(tk.Tk):
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

        # Generowanie odpowiedzi
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full_resp = self.tokenizer.decode(out[0], skip_special_tokens=True)
        answer = full_resp[len(prompt):].strip()
        self._add_message(answer)
        self._scroll_to_bottom()

    def _add_message(self, content):
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
    app = ChatApplication()
    app.mainloop()

# Wymagania: pip install torch transformers bitsandbytes sentencepiece protobuf
