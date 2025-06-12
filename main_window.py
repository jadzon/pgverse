import tkinter as tk
from tkinter import filedialog,messagebox
from bielik_app import ChatApplication
from integrated_rag_app import IntegratedRagApplication
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
        tk.Button(frame, text="Wczytaj plik", command=self.ocr_ksiazki, **btn_config).pack(pady=5)
        tk.Button(frame, text="Schematy", command=self.schemas, **btn_config).pack(pady=5)
        tk.Button(frame, text="Metryki Rag", command=self.check_rag_metrics, **btn_config).pack(pady=5)
        tk.Button(frame, text="Metryki Ocr", command=self.ocr_metryki, **btn_config).pack(pady=5)

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
        chat_window = IntegratedRagApplication()
        # Po zamknięciu okna Bielik, pokaż menu
        chat_window.protocol("WM_DELETE_WINDOW", lambda: self.on_bielik_close(chat_window))

    def ocr_ksiazki(self):
        """PDF file selection and OCR processing"""
        # Step 1: Select PDF file
        pdf_path = filedialog.askopenfilename(
            title="Wybierz plik PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), "ocr", "baza")
        )
        
        if not pdf_path:
            return  # User cancelled
            
        self.selected_pdf_path = pdf_path
        
        # Step 2: Select subject
        self.select_subject()
    
    def select_subject(self):
        """Open subject selection window"""
        if not self.selected_pdf_path:
            messagebox.showerror("Błąd", "Najpierw wybierz plik PDF")
            return
            
        # Create subject selection window
        subject_window = tk.Toplevel(self)
        subject_window.title("Wybierz przedmiot")
        subject_window.geometry("600x800")
        subject_window.configure(bg='#001F3F')
        subject_window.transient(self)
        subject_window.grab_set()
        
        # Get available subjects
        subjects = self.get_available_subjects()
        
        # Create scrollable frame
        canvas = tk.Canvas(subject_window, bg='#001F3F')
        scrollbar = tk.Scrollbar(subject_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#001F3F')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_label = tk.Label(
            scrollable_frame, 
            text="Wybierz przedmiot dla OCR:",
            font=("Arial", 14, "bold"),
            fg='white',
            bg='#001F3F'
        )
        title_label.pack(pady=20)
        
        # Subject buttons
        btn_config = dict(
            fg='white', bg='#003366', font=("Arial", 10), 
            width=50, height=2, wraplength=400
        )
        
        for subject in subjects:
            # Create a readable name from folder name
            display_name = subject.replace('_', ' ').title()
            
            btn = tk.Button(
                scrollable_frame,
                text=display_name,
                command=lambda s=subject: self.on_subject_selected(s, subject_window),
                **btn_config
            )
            btn.pack(pady=5, padx=20)
        
        # Cancel button
        cancel_btn = tk.Button(
            scrollable_frame,
            text="Anuluj",
            command=subject_window.destroy,
            fg='white', bg='#660000', font=("Arial", 12), width=20, height=2
        )
        cancel_btn.pack(pady=20)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def get_available_subjects(self):
        """Get list of available subjects from RAG/rag_codes/subjects/"""
        subjects_dir = os.path.join(os.path.dirname(__file__), "RAG", "rag_codes", "subjects")
        
        if not os.path.exists(subjects_dir):
            messagebox.showerror("Błąd", f"Katalog przedmiotów nie istnieje: {subjects_dir}")
            return []
        
        subjects = []
        for item in os.listdir(subjects_dir):
            item_path = os.path.join(subjects_dir, item)
            if os.path.isdir(item_path):
                subjects.append(item)
        
        return sorted(subjects)
    
    def on_subject_selected(self, subject, subject_window):
        """Handle subject selection and start OCR processing"""
        self.selected_subject = subject
        subject_window.destroy()
        
        # Show confirmation
        pdf_name = os.path.basename(self.selected_pdf_path)
        subject_display = subject.replace('_', ' ').title()
        
        confirm = messagebox.askyesno(
            "Potwierdzenie",
            f"Rozpocząć OCR dla:\n\nPlik: {pdf_name}\nPrzedmiot: {subject_display}\n\nTo może potrwać kilka minut."
        )
        if confirm:
            self.start_ocr_processing()
    
    def start_ocr_processing(self):
        """Start the OCR processing with selected PDF and subject"""
        if not self.selected_pdf_path or not self.selected_subject:
            messagebox.showerror("Błąd", "Brak wybranego pliku lub przedmiotu")
            return
        
        try:
            # Get paths
            pdf_name = os.path.splitext(os.path.basename(self.selected_pdf_path))[0]
            ocr_dir = os.path.join(os.path.dirname(__file__), "ocr", "scalanie_ocr")
              # Output directory should be the selected subject folder
            subject_dir = os.path.join(os.path.dirname(__file__), "RAG", "rag_codes", "subjects", self.selected_subject)
            
            # Create a subdirectory for this PDF in the subject folder (just PDF name, no prefix)
            output_dir = os.path.join(subject_dir, pdf_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Show processing message
            messagebox.showinfo(
                "Przetwarzanie", 
                f"OCR rozpoczęty. Wyniki będą zapisane w:\n{output_dir}\n\nSprawdź konsolę dla postępu."
            )
            
            # Change to OCR directory and run processing
            original_cwd = os.getcwd()
            os.chdir(ocr_dir)
            
            try:
                # Run updated mozg_ocr.py with arguments
                cmd = [
                    sys.executable, 
                    "mozg_ocr_updated.py",
                    "--input_pdf", self.selected_pdf_path,
                    "--subject", self.selected_subject,
                    "--output_dir", output_dir
                ]
                
                print(f"Executing: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode == 0:
                    messagebox.showinfo(
                        "Sukces", 
                        f"OCR zakończony pomyślnie!\n\nWyniki zapisane w folderze przedmiotu:\n{output_dir}"
                    )
                    print("OCR stdout:", result.stdout)
                else:
                    messagebox.showerror(
                        "Błąd OCR", 
                        f"OCR zakończony błędem:\n{result.stderr}"
                    )
                    print("OCR stderr:", result.stderr)
                    print("OCR stdout:", result.stdout)
                    
            finally:
                os.chdir(original_cwd)
                    
        except subprocess.TimeoutExpired:
            messagebox.showerror("Błąd", "OCR przekroczył limit czasu (1 godzina)")
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas OCR:\n{str(e)}")
            print(f"Exception details: {e}")
        
        # Reset selections
        self.selected_pdf_path = None
        self.selected_subject = None

    def ocr_metryki(self):
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