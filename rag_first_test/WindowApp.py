import tkinter as tk
from tkinter import END
from PIL import Image, ImageTk
import random

class ChatBot:
    def __init__(self):
        self.text_responses = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse."
        ]
        try:
            pil_image = Image.open('zdecie.png')
            self.image = ImageTk.PhotoImage(pil_image)
        except Exception as e:
            print(f"Błąd ładowania obrazu: {e}")
            self.image = None
    
    def get_response(self):
        if random.choice([True, False]):
            return random.choice(self.text_responses), 'text'
        else:
            return self.image, 'image' if self.image else ('Brak obrazu', 'text')

class ChatApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.bot = ChatBot()
        self.title("Chat Simulator")
        self.geometry("600x400")
        self.font_size = 12
        self.configure(bg='#001F3F')
        self.images = []
        self.create_widgets()
    
    def create_widgets(self):
        # GŁÓWNY FRAME NA CAŁE OKNO
        main_frame = tk.Frame(self, bg='#001F3F')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # CANVAS + SCROLLBAR
        self.chat_container = tk.Canvas(main_frame, bg='#001F3F', highlightthickness=0)
        self.chat_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0), pady=(10,0))

        self.scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=self.chat_container.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(10,60))  # pady dolne by nie nachodziło na input_frame

        self.chat_frame = tk.Frame(self.chat_container, bg='#001F3F')
        self.chat_container.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self.chat_container.configure(yscrollcommand=self.scrollbar.set)

        # Panel wprowadzania tekstu na dole
        input_frame = tk.Frame(self, bg='#001F3F')
        input_frame.pack(side="bottom", fill=tk.X, padx=10, pady=10)

        # Przyciski + i -
        zoom_in_button = tk.Button(
            input_frame, text="+", width=3, command=self.zoom_in,
            bg='#003366', fg='white', activebackground='#005599'
        )
        zoom_in_button.pack(side=tk.LEFT, padx=(0, 5))

        zoom_out_button = tk.Button(
            input_frame, text="-", width=3, command=self.zoom_out,
            bg='#003366', fg='white', activebackground='#005599'
        )
        zoom_out_button.pack(side=tk.LEFT, padx=(0, 10))

        self.user_input = tk.Entry(
            input_frame, font=("Arial", self.font_size),
            bg='#003366', fg='white', insertbackground='white'
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda e: self.send_message())

        tk.Button(
            input_frame, text="Wyślij", command=self.send_message, width=10,
            bg='#003366', fg='white', activebackground='#005599'
        ).pack(side=tk.RIGHT)

        # Obsługa dynamicznego rozmiaru
        self.chat_frame.bind("<Configure>", self._on_frame_configure)

    def zoom_in(self):
        self.font_size += 2
        self._update_font_size()

    def zoom_out(self):
        if self.font_size > 6:
            self.font_size -= 2
            self._update_font_size()

    def _update_font_size(self):
        for child in self.chat_frame.winfo_children():
            for subchild in child.winfo_children():
                if isinstance(subchild, tk.Label):
                    subchild.config(font=("Arial", self.font_size))
        self.user_input.config(font=("Arial", self.font_size))

    def send_message(self):
        message = self.user_input.get()
        if message:
            self._add_message(f"Ty: {message}", 'text')
            response, rtype = self.bot.get_response()
            self._add_message(response, rtype)
            self.user_input.delete(0, END)
            self._scroll_to_bottom()

    def _add_message(self, content, msg_type: str):
        frame = tk.Frame(self.chat_frame, bg='#001F3F')
        frame.pack(anchor='w', pady=2, fill=tk.X)

        if msg_type == 'text':
            label = tk.Label(
                frame, text=content, bg='#003366', fg='white',
                font=("Arial", self.font_size), wraplength=500, justify='left'
            )
            label.pack(padx=5, pady=2, anchor='w')
        else:
            label = tk.Label(frame, image=content, bg='#001F3F')
            label.image = content
            label.pack(anchor='w')
            self.images.append(content)

    def _on_frame_configure(self, event):
        self.chat_container.configure(scrollregion=self.chat_container.bbox("all"))

    def _scroll_to_bottom(self):
        self.chat_container.update_idletasks()
        self.chat_container.config(scrollregion=self.chat_container.bbox("all"))
        self.chat_container.yview_moveto(1.0)

if __name__ == "__main__":
    app = ChatApplication()
    app.mainloop()
