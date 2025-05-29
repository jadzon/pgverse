import tkinter as tk
from bielik_app import ChatApplication

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Menu główne")
        self.geometry("1000x1000")
        self.configure(bg='#001F3F')
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self, bg='#001F3F')
        frame.pack(expand=True)
        btn_config = dict(
            fg='white', bg='#003366', font=("Arial", 12), width=20, height=2
        )
        tk.Button(frame, text="RAG", command=self.rag_action, **btn_config).pack(pady=5)
        tk.Button(frame, text="Bielik", command=self.open_bielik, **btn_config).pack(pady=5)
        tk.Button(frame, text="Wczytaj plik", command=self.load_file, **btn_config).pack(pady=5)
        tk.Button(frame, text="Schematy", command=self.schemas, **btn_config).pack(pady=5)

    def rag_action(self):
        # TODO: dodać logikę RAG
        pass

    def load_file(self):
        # TODO: wczytywanie pliku
        pass

    def schemas(self):
        # TODO: obsługa schematów
        pass

    def open_bielik(self):
        # Ukryj okno menu
        self.withdraw()
        # Otwórz okno Bielik
        chat_window = ChatApplication()
        # Po zamknięciu okna Bielik, pokaż menu
        chat_window.protocol("WM_DELETE_WINDOW", lambda: self.on_bielik_close(chat_window))

    def on_bielik_close(self, chat_window):
        chat_window.destroy()
        self.deiconify()

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
