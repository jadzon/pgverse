import tkinter as tk
from tkinter import filedialog
from bielik_app import ChatApplication
from rag_app import RagApplication
import os
import sys
import subprocess
import tkinter.messagebox as mb
import enviromental_variables as ev

# Dodaj ścieżki do projektów schematics i charts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'schematics'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'charts'))

# Importy analizerów
try:
    from schematics import SchematicAnalyzer
except ImportError as e:
    print(f"Nie można zaimportować SchematicAnalyzer: {e}")
    SchematicAnalyzer = None

try:
    from charts import ChartAnalyzer
except ImportError as e:
    print(f"Nie można zaimportować ChartAnalyzer: {e}")
    ChartAnalyzer = None

def cleanup_bertscore():
        path = ev.JSON_PATH
        try:
            os.remove(path)
            print(f"Usunięto stary plik: {path}")
        except FileNotFoundError:
            pass

class SchemasWindow(tk.Tk):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.title("Wybór analizy")
        self.geometry("1000x1000")
        self.configure(bg='#001F3F')
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self, bg='#001F3F')
        frame.pack(expand=True)
        
        # Nagłówek
        title_label = tk.Label(frame, text="Wybierz typ analizy", 
                              fg='white', bg='#001F3F', 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        btn_config = dict(
            fg='white', bg='#003366', font=("Arial", 12), width=20, height=2
        )
        
        tk.Button(frame, text="Schematy", command=self.open_schematics, **btn_config).pack(pady=10)
        tk.Button(frame, text="Wykresy", command=self.open_charts, **btn_config).pack(pady=10)
        tk.Button(frame, text="Powrót", command=self.go_back, **btn_config).pack(pady=20)

    def open_schematics(self):
        # Otwórz dialog wyboru pliku dla schematów
        file_path = filedialog.askopenfilename(
            title="Wybierz plik schematu",
            filetypes=[
                ("Pliki obrazów", "*.png *.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        
        if file_path:
            if SchematicAnalyzer is None:
                mb.showerror("Błąd", "SchematicAnalyzer nie jest dostępny. Sprawdź czy projekt schematics jest w odpowiednim miejscu.")
                return
            
            try:
                # Utwórz folder wyników
                results_folder = "schematy_results"
                
                # Inicjalizuj analyzer z odpowiednimi parametrami
                analyzer = SchematicAnalyzer(
                    model_path=os.path.join("..", "..", "schematics", "block_detector", "models", "handwritten.pt"),
                    results_folder=results_folder,
                    preprocess_enabled=True,
                    text_detection_enabled=True
                )
                
                mb.showinfo("Analiza schematów", "Rozpoczynam analizę schematu...\nMoże to potrwać kilka minut.")
                
                # Przeprowadź analizę
                result = analyzer.analyze(image_path=file_path)
                
                if result:
                    mb.showinfo("Sukces", f"Analiza schematu zakończona!\n\nWyniki zapisane w folderze: {results_folder}")
                else:
                    mb.showwarning("Ostrzeżenie", "Analiza zakończona, ale nie wykryto komponentów elektronicznych.")
                    
            except Exception as e:
                mb.showerror("Błąd", f"Wystąpił błąd podczas analizy schematu:\n{str(e)}")
                print(f"Błąd analizy schematu: {e}")

    def open_charts(self):
        # Otwórz dialog wyboru pliku dla wykresów
        file_path = filedialog.askopenfilename(
            title="Wybierz plik wykresu",
            filetypes=[
                ("Pliki obrazów", "*.png *.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        
        if file_path:
            if ChartAnalyzer is None:
                mb.showerror("Błąd", "ChartAnalyzer nie jest dostępny. Sprawdź czy projekt charts jest w odpowiednim miejscu.")
                return
            
            try:
                # Utwórz folder wyników
                results_folder = "wykresy_results"
                os.makedirs(results_folder, exist_ok=True)
                
                # Inicjalizuj analyzer
                analyzer = ChartAnalyzer(chart_path=file_path)
                
                mb.showinfo("Analiza wykresów", "Rozpoczynam analizę wykresu...\nMoże to potrwać kilka minut.")
                
                # Przeprowadź analizę
                result = analyzer.analyze()
                
                # Przenieś wyniki do naszego folderu
                source_results = "results"
                if os.path.exists(source_results):
                    import shutil
                    for item in os.listdir(source_results):
                        src = os.path.join(source_results, item)
                        dst = os.path.join(results_folder, item)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src, dst)
                
                mb.showinfo("Sukces", f"Analiza wykresu zakończona!\n\nWyniki zapisane w folderze: {results_folder}")
                    
            except Exception as e:
                mb.showerror("Błąd", f"Wystąpił błąd podczas analizy wykresu:\n{str(e)}")
                print(f"Błąd analizy wykresu: {e}")

    def go_back(self):
        self.destroy()
        self.parent_window.deiconify()

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
        # Ukryj okno menu
        self.withdraw()
        # Otwórz okno wyboru schematów
        schemas_window = SchemasWindow(self)
        # Po zamknięciu okna, pokaż menu
        schemas_window.protocol("WM_DELETE_WINDOW", lambda: self.on_schemas_close(schemas_window))

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

    def on_schemas_close(self, schemas_window):
        schemas_window.destroy()
        self.deiconify()

if __name__ == "__main__":
    #cleanup_bertscore()
    app = MainWindow()
    app.mainloop()
