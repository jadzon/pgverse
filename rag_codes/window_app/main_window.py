import tkinter as tk
from bielik_app import ChatApplication
from rag_app import RagApplication
import os
import sys
import subprocess
import tkinter.messagebox as mb
import enviromental_variables as ev

def cleanup_bertscore():
        path = ev.JSON_PATH
        try:
            os.remove(path)
            print(f"Usunięto stary plik: {path}")
        except FileNotFoundError:
            pass

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
        tk.Button(frame, text="Metryki Rag", command=self.check_rag_metrics, **btn_config).pack(pady=5)

    def check_rag_metrics(self):
        script = "rag_metrics/bertscore_graph.py"
        if not os.path.exists(script):
            mb.showerror("Błąd", f"Nie znaleziono pliku: {script}")
            return
        try:
            # używamy tego samego interpretera Pythona
            subprocess.Popen([sys.executable, script])
        except Exception as e:
            mb.showerror("Błąd", f"Nie udało się uruchomić skryptu {script}:\n{e}")

    def rag_action(self):
        # Ukryj okno menu
        self.withdraw()
        # Otwórz okno Bielik
        chat_window = RagApplication()
        # Po zamknięciu okna Bielik, pokaż menu
        chat_window.protocol("WM_DELETE_WINDOW", lambda: self.on_bielik_close(chat_window))

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
    #cleanup_bertscore()
    app = MainWindow()
    app.mainloop()
