import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import json
import threading
import hashlib
import torch

# Importy RAG functions
from rag_codes.rag_functions.metadata_caption import ImageTextProcessor, ImageContextFilter
from rag_codes.rag_functions.embeddings import CLIPEmbedder
from rag_codes.rag_functions.chunker import TextChunker
from rag_codes.rag_functions.graph import (
    TextGraphBuilder, Neo4jConnector, HybridTextRetriever, 
    GraphPruner, LearningPatternTracker
)

class SubjectSelectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Wybór przedmiotów")
        self.root.geometry("800x600")
        
        # Ścieżki
        self.base_path = Path(__file__).parent.resolve()
        self.subjects_path = self.base_path / "rag_codes" / "subjects"
        
        # Dane aplikacji
        self.subject_vars = {}
        self.subject_checkboxes = {}
        self.temp_sources_configs = {}
        
        # Dostępne typy źródeł
        self.source_types = [
            "wikipedia", "książka", "artykuł_naukowy", "blog",
            "forum", "social_media", "news", "unknown"
        ]
        
        # Mapowanie folderów na typy danych
        self.folder_type_mapping = {
            "figure": "image", "figures": "image", "figury": "image",
            "obrazy": "image", "zdjecia": "image", "tabele": "table",
            "tables": "table", "tabela": "table", "table": "table",
            "wzory": "formula", "formulas": "formula", "formula": "formula",
            "wzor": "formula", "equations": "formula", "rownania": "formula"
        }
        
        self.create_widgets()
        self.load_subjects()
        
    def create_widgets(self):
        """Tworzy interfejs głównego okna"""
        # Tytuł
        title_label = tk.Label(self.root, text="Wybierz przedmioty do wczytania", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Informacja o ścieżce
        path_label = tk.Label(self.root, text=f"Ścieżka: {self.subjects_path}", 
                             font=("Arial", 8), fg="gray")
        path_label.pack(pady=5)
        
        # Informacja o wymaganiu
        info_label = tk.Label(self.root, 
                             text="Aby zaznaczyć przedmiot, musisz najpierw przypisać źródła do folderów", 
                             font=("Arial", 9), fg="blue")
        info_label.pack(pady=5)
        
        # Frame z scrollbarem dla listy przedmiotów
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 5))  # Zmniejszony pady bottom
        
        # Scrollbar
        scrollbar = tk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas dla przewijania
        canvas = tk.Canvas(main_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        # Frame wewnętrzny dla checkboxów
        self.checkbox_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")
        
        # Aktualizacja scroll region - POPRAWIONE
        def configure_scroll_region(event):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except tk.TclError:
                # Widget został zniszczony
                pass
        
        self.checkbox_frame.bind("<Configure>", configure_scroll_region)
        
        # Przyciski kontrolne (Zaznacz/Odznacz wszystko)
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)  # Zmniejszony padding
        
        tk.Button(button_frame, text="Zaznacz wszystko", 
                 command=self.select_all).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Odznacz wszystko", 
                 command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        
        # Frame dla przycisków głównych - ZAMKNIJ i DALEJ
        main_buttons_frame = tk.Frame(self.root)
        main_buttons_frame.pack(pady=10, side=tk.BOTTOM)  # Dodane side=tk.BOTTOM
        
        # Przycisk ZAMKNIJ - zamyka program
        tk.Button(main_buttons_frame, 
                 text="ZAMKNIJ", 
                 command=self.close_application,
                 bg="red", 
                 fg="white",
                 font=("Arial", 12, "bold"),
                 width=12,
                 height=2).pack(side=tk.LEFT, padx=20)
        
        # Przycisk DALEJ - zapisuje JSONy i kontynuuje
        tk.Button(main_buttons_frame, 
                 text="DALEJ", 
                 command=self.save_and_proceed,
                 bg="green", 
                 fg="white",
                 font=("Arial", 12, "bold"),
                 width=12,
                 height=2).pack(side=tk.LEFT, padx=20)

        # NOWY: Przycisk Zarządzanie bazą danych
        tk.Button(main_buttons_frame,
                 text="Zarządzanie bazą danych",
                 command=lambda: self.open_graph_management_window(list(self.subject_vars.keys())),
                 bg="blue",
                 fg="white",
                 font=("Arial", 12, "bold"),
                 width=20,
                 height=2).pack(side=tk.LEFT, padx=20)

    def load_subjects(self):
        """Wczytuje listę przedmiotów i tworzy checkboxy + przyciski Źródła"""
        if not self.subjects_path.exists():
            messagebox.showerror("Błąd", f"Nie znaleziono folderu {self.subjects_path}")
            return

        for item in sorted(self.subjects_path.iterdir()):
            if not item.is_dir() or item.name == "__pycache__":
                continue

            name = item.name
            var = tk.BooleanVar(value=False)

            # Frame na checkbox + przycisk Źródła
            row = tk.Frame(self.checkbox_frame)
            row.pack(fill=tk.X, pady=2)

            cb = tk.Checkbutton(row, text=name, variable=var)
            cb.pack(side=tk.LEFT)

            src_btn = tk.Button(row,
                                text="Źródła",
                                command=lambda n=name: self.open_source_config(n),
                                width=8)
            src_btn.pack(side=tk.LEFT, padx=5)

            self.subject_vars[name] = var
            self.subject_checkboxes[name] = cb

    def select_all(self):
        """Zaznacza wszystkie checkboxy"""
        for var in self.subject_vars.values():
            var.set(True)

    def deselect_all(self):
        """Odznacza wszystkie checkboxy"""
        for var in self.subject_vars.values():
            var.set(False)

    def close_application(self):
        """Zamyka aplikację z potwierdzeniem"""
        message = "Czy na pewno chcesz zamknąć aplikację?"
        if self.temp_sources_configs:
            message += "\nMasz niezapisane zmiany w źródłach, które zostaną utracone!"
        
        if messagebox.askyesno("Potwierdzenie zamknięcia", message):
            self.root.destroy()

    def save_and_proceed(self):
        """Zapisuje konfiguracje i przechodzi dalej"""
        selected_subjects = [subject for subject, var in self.subject_vars.items() if var.get()]
        
        if not selected_subjects:
            messagebox.showwarning("Brak wyboru", 
                                  "Nie wybrano żadnego przedmiotu!\n"
                                  "Zaznacz przynajmniej jeden przedmiot przed kontynuowaniem.")
            return

        # Sprawdzenie źródeł (placeholder - zawsze True)
        incomplete_subjects = [s for s in selected_subjects if not self.check_sources_complete(s)]
        
        if incomplete_subjects:
            messagebox.showerror("Błąd - nieprzypisane źródła", 
                f"Następujące przedmioty mają nieprzypisane źródła:\n\n" + 
                "\n".join(f"• {subject}" for subject in incomplete_subjects))
            return

        # Zapisywanie konfiguracji
        saved_count = 0
        failed_subjects = []
        updated_subjects = []
        
        for subject_name in selected_subjects:
            subject_path = self.subjects_path / subject_name
            
            if subject_name in self.temp_sources_configs:
                if self.save_sources_config(subject_path, self.temp_sources_configs[subject_name]):
                    saved_count += 1
                    updated_subjects.append(subject_name)
                else:
                    failed_subjects.append(subject_name)
        
        if failed_subjects:
            messagebox.showerror("Błąd zapisywania", 
                f"Nie udało się zapisać konfiguracji dla przedmiotów:\n\n" + 
                "\n".join(f"• {subject}" for subject in failed_subjects))
            return

        # Komunikat sukcesu
        success_message = f"🎯 SUKCES!\n\nWybrano {len(selected_subjects)} przedmiotów:\n"
        success_message += "\n".join(f"• {subject}" for subject in selected_subjects)
        
        if saved_count > 0:
            success_message += f"\n\n💾 Zapisano {saved_count} nowych konfiguracji źródeł"
        
        success_message += "\n\n🚀 Konfiguracja zakończona pomyślnie!"
        
        messagebox.showinfo("Konfiguracja zakończona", success_message)
        self.temp_sources_configs.clear()
        self.open_processing_window(selected_subjects)

    def check_sources_complete(self, subject_name):
        """Sprawdza czy przedmiot ma kompletne źródła (placeholder)"""
        return True
    
    def save_sources_config(self, subject_path, config):
        """Zapisuje konfigurację źródeł do pliku JSON"""
        try:
            config_file = subject_path / "sources_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def open_source_config(self, subject_name):
        """Okno konfiguracji źródeł dla podfolderów wybranego przedmiotu"""
        subject_path = self.subjects_path / subject_name
        existing = self.temp_sources_configs.get(subject_name, {})

        win = tk.Toplevel(self.root)
        win.title(f"Źródła: {subject_name}")
        win.geometry("400x300")
        win.grab_set()

        vars_map = {}
        tk.Label(win, text=f"Typ źródła dla folderów {subject_name}",
                 font=("Arial", 12, "bold")).pack(pady=10)

        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)

        for sub in sorted(subject_path.iterdir()):
            if not sub.is_dir() or sub.name == "__pycache__":
                continue
            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=sub.name, width=15, anchor="w").pack(side=tk.LEFT)

            var = tk.StringVar(value=existing.get(sub.name, "unknown"))
            menu = ttk.OptionMenu(row, var, var.get(), *self.source_types)
            menu.pack(side=tk.LEFT, padx=5)
            vars_map[sub.name] = var

        def save_and_close():
            self.temp_sources_configs[subject_name] = {k: v.get() for k, v in vars_map.items()}
            win.destroy()

        tk.Button(win, text="Zapisz", bg="green", fg="white",
                  command=save_and_close).pack(pady=10)

    def open_processing_window(self, selected_subjects):
        """Otwiera okno przetwarzania"""
        processing_window = tk.Toplevel(self.root)
        processing_window.title("Przetwarzanie podfolderów")
        processing_window.geometry("900x700")
        processing_window.grab_set()
        
        # Tytuł
        tk.Label(processing_window, text="Przetwarzanie podfolderów do formatu JSON", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Info o przedmiatch
        tk.Label(processing_window, 
                text=f"Wybrane przedmioty ({len(selected_subjects)}): " + ", ".join(selected_subjects), 
                font=("Arial", 10), wraplength=800).pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(processing_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Log area
        log_frame = tk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(log_frame, text="Log przetwarzania:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        text_frame = tk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        log_text = tk.Text(text_frame, yscrollcommand=scrollbar.set, 
                          font=("Consolas", 9), wrap=tk.WORD)
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=log_text.yview)
        
        # Progress bar
        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(progress_frame, text="Postęp:", font=("Arial", 10, "bold")).pack(anchor="w")
        progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=2)
        
        progress_label = tk.Label(progress_frame, text="Gotowy do rozpoczęcia", font=("Arial", 9))
        progress_label.pack(anchor="w")
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        start_btn = tk.Button(button_frame, text="ROZPOCZNIJ PRZETWARZANIE", 
                             command=lambda: self.start_processing(selected_subjects, log_text, 
                                                                  progress_bar, progress_label, 
                                                                  start_btn, close_btn),
                             bg="green", fg="white", font=("Arial", 12, "bold"),
                             width=25, height=2)
        start_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = tk.Button(button_frame, text="ZAMKNIJ", 
                             command=processing_window.destroy,
                             bg="red", fg="white", font=("Arial", 12, "bold"),
                             width=15, height=2)
        close_btn.pack(side=tk.LEFT, padx=10)

    def start_processing(self, selected_subjects, log_text, progress_bar, progress_label, start_btn, close_btn):
        """Rozpoczyna przetwarzanie w osobnym wątku"""
        start_btn.config(state=tk.DISABLED)
        
        def process_in_thread():
            self.process_all_subjects(selected_subjects, log_text, progress_bar, progress_label)
            start_btn.config(state=tk.NORMAL, text="ZAKOŃCZONO", bg="gray")
        
        processing_thread = threading.Thread(target=process_in_thread)
        processing_thread.daemon = True
        processing_thread.start()

    def log_message(self, log_text, message):
        """Dodaje wiadomość do log widget"""
        try:
            log_text.insert(tk.END, f"{message}\n")
            log_text.see(tk.END)
            log_text.update_idletasks()
        except tk.TclError:
            pass

    def update_progress(self, progress_bar, progress_label, current, total, current_task=""):
        """Aktualizuje progress bar"""
        try:
            percentage = (current / total) * 100 if total > 0 else 0
            progress_bar['value'] = percentage
            progress_label.config(text=f"Postęp: {current}/{total} ({percentage:.1f}%) - {current_task}")
            progress_bar.update_idletasks()
            progress_label.update_idletasks()
        except tk.TclError:
            pass

    def normalize_path_separators(self, path):
        """
        Normalizuje separatory ścieżek do pojedynczych slashów
        """
        # Zamień podwójne backslashe na pojedyncze slashe
        normalized = str(path).replace('\\\\', '/').replace('\\', '/')
        return normalized

    def convert_to_relative_path(self, absolute_path):
        """
        Konwertuje ścieżkę absolutną na względną od folderu pgverse
        """
        try:
            path_obj = Path(absolute_path)
            
            # Znajdź część ścieżki zaczynającą się od pgverse
            path_parts = path_obj.parts
            pgverse_index = -1
            
            for i, part in enumerate(path_parts):
                if part.lower() == "pgverse":
                    pgverse_index = i
                    break
            
            if pgverse_index >= 0:
                # Zbuduj ścieżkę od pgverse (włącznie z pgverse)
                relative_parts = path_parts[pgverse_index:]
                # POPRAWKA: Użyj backslashes dla Windows
                relative_path = '\\'.join(relative_parts)  # Używaj backslashes
                return relative_path
            
            # Jeśli nie znaleziono pgverse, zwróć oryginalną ścieżkę
            return str(absolute_path)
            
        except Exception as e:
            print(f"DEBUG convert_to_relative_path error for {absolute_path}: {e}")
            return str(absolute_path)

    def process_all_subjects(self, selected_subjects, log_text, progress_bar, progress_label):
        """Przetwarza wszystkie wybrane przedmioty"""
        self.log_message(log_text, "=== ROZPOCZĘCIE PRZETWARZANIA PRZEDMIOTÓW ===")
        
        # Inicjalizacja procesorów
        try:
            self.log_message(log_text, "Inicjalizacja procesorów...")
            processor = ImageTextProcessor()
            filter_processor = ImageContextFilter()
            self.log_message(log_text, "✓ Procesory zainicjalizowane pomyślnie")
        except Exception as e:
            self.log_message(log_text, f"✗ Błąd inicjalizacji procesorów: {e}")
            return
        
        # Liczenie folderów do przetworzenia - ZMIANA: detekcje zamiast rezultaty
        total_ocr_folders = 0
        subject_ocr_folders = {}
        
        for subject_name in selected_subjects:
            subject_path = self.subjects_path / subject_name
            ocr_folders = [item for item in subject_path.iterdir() 
                          if item.is_dir() and item.name != "__pycache__"]
            subject_ocr_folders[subject_name] = ocr_folders
            total_ocr_folders += len(ocr_folders)
        
        self.log_message(log_text, f"Znaleziono {total_ocr_folders} folderów OCR do przetworzenia")
        
        if total_ocr_folders == 0:
            self.log_message(log_text, "Brak folderów OCR do przetworzenia!")
            return
        
        processed_count = 0
        success_count = 0
        error_count = 0
        
        # Przetwarzanie
        for subject_name in selected_subjects:
            self.log_message(log_text, f"\n--- PRZETWARZANIE PRZEDMIOTU: {subject_name} ---")
            
            ocr_folders = subject_ocr_folders[subject_name]
            
            for ocr_folder in ocr_folders:
                processed_count += 1
                self.update_progress(progress_bar, progress_label, processed_count, total_ocr_folders, 
                                   f"{subject_name}/{ocr_folder.name}")
                
                self.log_message(log_text, f"\nPrzetwarzanie folderu OCR: {subject_name}/{ocr_folder.name}")
                
                # Definiuj ścieżkę do folderu detekcje
                detekcje_path = ocr_folder / "detekcje"
                
                # Sprawdź czy folder detekcje istnieje
                if not detekcje_path.exists():
                    self.log_message(log_text, f"  ⚠ Brak folderu 'detekcje' w {ocr_folder.name}")
                    continue
                
                # Przetwarzanie tylko pliku txt o nazwie podfolderu
                expected_txt_file = detekcje_path / f"{ocr_folder.name}.txt"
                if not expected_txt_file.exists():
                    self.log_message(log_text, f"  ⚠ Brak pliku {ocr_folder.name}.txt w {ocr_folder.name}/detekcje")
                    continue
                
                txt_files = [expected_txt_file]  # Lista z jednym plikiem
                
                # Przetwarzanie plików txt
                for txt_file in txt_files:
                    try:
                        self.log_message(log_text, f"  📄 Przetwarzanie pliku: {txt_file.name}")
                        
                        # KROK 1: Przetwórz plik i pobierz texts
                        texts = processor.process_file(str(txt_file))
                        self.log_message(log_text, f"    📋 Znaleziono {len(texts)} elementów do przetworzenia")
                        
                        # KROK 2: Utwórz JSON z kontekstem obrazów (już przefiltrowany!)
                        json_data = processor.get_images_with_context_json(texts)
                        
                        # NOWE: Normalizuj ścieżki w JSON przed zapisem
                        if json_data:
                            normalized_json_data = []
                            for item in json_data:
                                normalized_item = {}
                                for image_path, context_texts in item.items():
                                    # Normalizuj ścieżkę do formatu z pojedynczymi slashami
                                    normalized_path = self.normalize_path_separators(image_path)
                                    normalized_item[normalized_path] = context_texts
                                normalized_json_data.append(normalized_item)
                            json_data = normalized_json_data
                        
                        if not json_data:
                            self.log_message(log_text, f"    ⚠ Brak istniejących obrazów w pliku {txt_file.name}")
                        else:
                            self.log_message(log_text, f"    ✓ Znaleziono {len(json_data)} istniejących obrazów")
                            
                            # Zapisz JSON bezpośrednio (bez dodatkowego filtrowania)
                            json_output_file = detekcje_path / f"{txt_file.stem}_filtered_context.json"
                            
                            try:
                                with open(json_output_file, 'w', encoding='utf-8') as f:
                                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                                self.log_message(log_text, f"    ✓ Zapisano JSON: {json_output_file.name}")
                                json_result = True
                            except Exception as e:
                                self.log_message(log_text, f"    ✗ Błąd zapisywania JSON: {e}")
                                json_result = False
                        
                        # KROK 3: Utwórz plik TXT z samymi chunkami
                        chunks_output_file = detekcje_path / f"{txt_file.stem}_chunks.txt"
                        chunks_result = processor.create_output_txt_chunks_only(texts, str(chunks_output_file))
                        
                        if chunks_result:
                            self.log_message(log_text, f"    ✓ Zapisano chunks TXT: {chunks_output_file.name}")
                        else:
                            self.log_message(log_text, f"    ✗ Błąd zapisywania chunks TXT: {chunks_output_file.name}")
                        
                        # KROK 4: Utwórz plik TXT z base64
                        base64_output_file = detekcje_path / f"{txt_file.stem}_base64.txt"
                        base64_result = processor.create_output_txt_with_base64(texts, str(base64_output_file))
                        
                        if base64_result:
                            self.log_message(log_text, f"    ✓ Zapisano base64 TXT: {base64_output_file.name}")
                        else:
                            self.log_message(log_text, f"    ✗ Błąd zapisywania base64 TXT: {base64_output_file.name}")
                        
                        # Zlicz sukces jeśli przynajmniej jeden plik został utworzony
                        if (json_data and json_result) or chunks_result or base64_result:
                            success_count += 1
                        else:
                            error_count += 1
                        
                    except Exception as e:
                        self.log_message(log_text, f"    ✗ Błąd przetwarzania {txt_file.name}: {e}")
                        error_count += 1
        
        # Podsumowanie
        self.log_message(log_text, f"\n=== PODSUMOWANIE PRZETWARZANIA ===")
        self.log_message(log_text, f"Przetworzono przedmiotów: {len(selected_subjects)}")
        self.log_message(log_text, f"Przetworzono folderów OCR: {processed_count}")
        self.log_message(log_text, f"Pomyślnie przetworzone pliki: {success_count}")
        self.log_message(log_text, f"Błędy: {error_count}")
        
        if error_count == 0:
            self.log_message(log_text, "🎉 WSZYSTKIE PLIKI PRZETWORZONE POMYŚLNIE!")
            self.log_message(log_text, "📋 Utworzono dla każdego pliku:")
            self.log_message(log_text, "  • JSON z kontekstem obrazów (*_filtered_context.json)")
            self.log_message(log_text, "  • TXT z chunkami tekstowymi (*_chunks.txt)")
            self.log_message(log_text, "  • TXT z obrazami base64 (*_base64.txt)")
            
            result = messagebox.askyesno("Zarządzanie grafem Neo4j", 
                "Przetwarzanie zakończone pomyślnie!\n\n"
                "Czy chcesz otworzyć okno zarządzania grafem Neo4j\n"
                "do tworzenia relacji między chunkami tekstu?")
            if result:
                self.open_graph_management_window(selected_subjects)
        else:
            self.log_message(log_text, f"⚠ ZAKOŃCZONO Z {error_count} BŁĘDAMI")
        
        self.update_progress(progress_bar, progress_label, total_ocr_folders, total_ocr_folders, "Zakończono")

    def open_graph_management_window(self, selected_subjects):
        """Otwiera okno zarządzania grafem Neo4j"""
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Zarządzanie grafem Neo4j")
        graph_window.geometry("1000x800")
        graph_window.grab_set()
        
        # Tytuł
        tk.Label(graph_window, text="Zarządzanie grafem Neo4j - Tworzenie relacji między chunkami", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Info o przedmiatch
        tk.Label(graph_window, text=f"Dostępne przedmioty: {', '.join(selected_subjects)}", 
                font=("Arial", 10), wraplength=900).pack(pady=5)
        
        # NOWE: Frame wyboru przedmiotów
        subject_selection_frame = tk.LabelFrame(graph_window, text="Wybór przedmiotów do załadowania", 
                                               font=("Arial", 10, "bold"))
        subject_selection_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Checkboxy dla przedmiotów
        subject_vars = {}
        subjects_frame = tk.Frame(subject_selection_frame)
        subjects_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Pierwszy rząd - checkboxy przedmiotów
        subjects_row1 = tk.Frame(subjects_frame)
        subjects_row1.pack(fill=tk.X, pady=2)
        
        for i, subject in enumerate(selected_subjects):
            var = tk.BooleanVar(value=True)  # Domyślnie zaznaczone
            cb = tk.Checkbutton(subjects_row1, text=subject, variable=var)
            cb.pack(side=tk.LEFT, padx=10)
            subject_vars[subject] = var
            
            # Nowy rząd co 4 przedmioty dla lepszego układu
            if (i + 1) % 4 == 0 and i < len(selected_subjects) - 1:
                subjects_row1 = tk.Frame(subjects_frame)
                subjects_row1.pack(fill=tk.X, pady=2)
        
        # Przyciski zaznacz/odznacz wszystko
        selection_buttons_frame = tk.Frame(subjects_frame)
        selection_buttons_frame.pack(fill=tk.X, pady=5)
        
        def select_all_subjects():
            for var in subject_vars.values():
                var.set(True)
        
        def deselect_all_subjects():
            for var in subject_vars.values():
                var.set(False)
        
        tk.Button(selection_buttons_frame, text="Zaznacz wszystkie", 
                 command=select_all_subjects).pack(side=tk.LEFT, padx=5)
        tk.Button(selection_buttons_frame, text="Odznacz wszystkie", 
                 command=deselect_all_subjects).pack(side=tk.LEFT, padx=5)
        
        # Konfiguracja połączenia
        connection_frame = tk.LabelFrame(graph_window, text="Konfiguracja połączenia Neo4j", 
                                        font=("Arial", 10, "bold"))
        connection_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Pola połączenia
        tk.Label(connection_frame, text="URI:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        uri_var = tk.StringVar(value="neo4j+s://335a260d.databases.neo4j.io")
        tk.Entry(connection_frame, textvariable=uri_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(connection_frame, text="User:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        user_var = tk.StringVar(value="neo4j")
        tk.Entry(connection_frame, textvariable=user_var, width=15).grid(row=0, column=3, padx=5, pady=5)
        
        tk.Label(connection_frame, text="Password:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        password_var = tk.StringVar(value="4RMc8un8Yjzx9oy3_l2fDw5pNbuKuNdGjLHFI4a_EEU")
        tk.Entry(connection_frame, textvariable=password_var, show="*", width=15).grid(row=1, column=1, padx=5, pady=5)
        
        # Status połączenia
        status_var = tk.StringVar(value="Nie połączono")
        status_label = tk.Label(connection_frame, textvariable=status_var, fg="red", font=("Arial", 9))
        status_label.grid(row=1, column=2, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Logi
        log_frame = tk.LabelFrame(graph_window, text="Log operacji", font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        text_frame = tk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        graph_log_text = tk.Text(text_frame, yscrollcommand=scrollbar.set, 
                                font=("Consolas", 9), wrap=tk.WORD, height=15)
        graph_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=graph_log_text.yview)
        
        # Zmienne dla komponentów grafu
        neo4j_connector = None
        graph_builder = None
        
        def log_graph_message(message):
            """Dodaje wiadomość do log widget grafu"""
            try:
                graph_log_text.insert(tk.END, f"{message}\n")
                graph_log_text.see(tk.END)
                graph_log_text.update_idletasks()
            except tk.TclError:
                pass
        
        def test_connection():
            """Testuje połączenie z Neo4j"""
            try:
                connector = Neo4jConnector(uri_var.get(), user_var.get(), password_var.get())
                with connector.get_driver().session() as session:
                    session.run("RETURN 1")
                connector.close()
                status_var.set("✓ Połączenie prawidłowe")
                status_label.config(fg="green")
                connect_btn.config(state=tk.NORMAL)
            except Exception as e:
                status_var.set(f"✗ Błąd: {str(e)[:50]}...")
                status_label.config(fg="red")
                connect_btn.config(state=tk.DISABLED)
        
        def connect_to_neo4j():
            """Łączy się z Neo4j"""
            nonlocal neo4j_connector, graph_builder
            
            try:
                log_graph_message("🔄 Łączenie z Neo4j...")
                neo4j_connector = Neo4jConnector(uri_var.get(), user_var.get(), password_var.get())
                
                log_graph_message("🔄 Inicjalizacja komponentów grafu...")
                graph_builder = TextGraphBuilder(neo4j_connector, similarity_threshold=0.8)
                
                log_graph_message("✓ Pomyślnie połączono z Neo4j")
                
                for btn in operation_buttons:
                    btn.config(state=tk.NORMAL)
                
                connect_btn.config(state=tk.DISABLED, text="Połączono")
                disconnect_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                log_graph_message(f"✗ Błąd połączenia: {e}")
        
        def disconnect_from_neo4j():
            """Rozłącza z Neo4j"""
            nonlocal neo4j_connector, graph_builder
            
            try:
                if neo4j_connector:
                    neo4j_connector.close()
                if graph_builder:
                    graph_builder.close()
                
                neo4j_connector = None
                graph_builder = None
                
                log_graph_message("✓ Rozłączono z Neo4j")
                
                for btn in operation_buttons:
                    btn.config(state=tk.DISABLED)
                
                connect_btn.config(state=tk.NORMAL, text="Połącz z Neo4j")
                disconnect_btn.config(state=tk.DISABLED)
                
            except Exception as e:
                log_graph_message(f"✗ Błąd rozłączania: {e}")
        
        # Funkcje operacji na grafie (teraz w wątkach)
        def create_relations():
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return

            def worker():
                try:
                    log_graph_message("🔄 Tworzenie relacji podobieństwa w grafie…")
                    graph_builder.create_relations()
                    log_graph_message("✓ Relacje podobieństwa utworzone pomyślnie")
                except Exception as e:
                    log_graph_message(f"✗ Błąd tworzenia relacji: {e}")

            threading.Thread(target=worker, daemon=True).start()

        def show_statistics():
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            try:
                log_graph_message("🔄 Pobieranie statystyk grafu…")
                stats = graph_builder.analyze_learning_patterns()
                log_graph_message("=== STATYSTYKI GRAFU ===")
                log_graph_message(f"Węzły tekstowe: {stats['node_statistics'].get('total_nodes', 0)}")
                log_graph_message(f"Wszystkich relacji: {stats['relation_statistics'].get('total_relations', 0)}")
            except Exception as e:
                log_graph_message(f"✗ Błąd pobierania statystyk: {e}")
        
        def clear_all_chunks():
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            if messagebox.askyesno("Potwierdzenie", 
                "Czy na pewno chcesz usunąć WSZYSTKIE CHUNKI z grafu?\nTa operacja jest nieodwracalna!"):
                try:
                    log_graph_message("🔄 Usuwanie wszystkich chunków...")
                    with neo4j_connector.get_driver().session() as session:
                        result = session.run("MATCH (n:TextNode) DETACH DELETE n")
                        deleted_count = result.consume().counters.nodes_deleted
                    log_graph_message(f"✓ Usunięto {deleted_count} chunków tekstowych")
                except Exception as e:
                    log_graph_message(f"✗ Błąd usuwania chunków: {e}")

        def clear_entire_database():
            """Czyści całą bazę danych Neo4j"""
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            # Podwójne potwierdzenie dla tej krytycznej operacji
            if messagebox.askyesno("⚠️ UWAGA - NIEBEZPIECZNA OPERACJA", 
                "Czy na pewno chcesz usunąć CAŁĄ BAZĘ DANYCH?\n\n"
                "Ta operacja:\n"
                "• Usunie WSZYSTKIE węzły (Chunk, TextNode, itp.)\n"
                "• Usunie WSZYSTKIE relacje\n"
                "• Wyczyści całą bazę Neo4j\n"
                "• Jest NIEODWRACALNA!\n\n"
                "Czy jesteś absolutnie pewien?"):
                
                # Drugie potwierdzenie
                confirm = messagebox.askquestion("🚨 OSTATNIE POTWIERDZENIE", 
                    "To jest ostatnia szansa na anulowanie!\n\n"
                    "Kliknij 'yes' aby BEZPOWROTNIE USUNĄĆ całą bazę danych\n"
                    "Kliknij 'no' aby anulować operację",
                    icon='warning')
                
                if confirm == 'yes':
                    try:
                        log_graph_message("🚨 ROZPOCZYNAM CZYSZCZENIE CAŁEJ BAZY DANYCH...")
                        
                        with neo4j_connector.get_driver().session() as session:
                            # Najpierw policz co będzie usunięte
                            count_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                            total_nodes = count_result.single()["total_nodes"]
                            
                            rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relations")
                            total_relations = rel_count_result.single()["total_relations"]
                            
                            log_graph_message(f"🔄 Znaleziono {total_nodes} węzłów i {total_relations} relacji do usunięcia...")
                            
                            # Usuń wszystkie relacje
                            log_graph_message("🔄 Usuwanie wszystkich relacji...")
                            rel_result = session.run("MATCH ()-[r]->() DELETE r")
                            deleted_relations = rel_result.consume().counters.relationships_deleted
                            
                            # Usuń wszystkie węzły
                            log_graph_message("🔄 Usuwanie wszystkich węzłów...")
                            node_result = session.run("MATCH (n) DELETE n")
                            deleted_nodes = node_result.consume().counters.nodes_deleted
                            
                            # Weryfikuj że baza jest pusta
                            verify_result = session.run("MATCH (n) RETURN count(n) as remaining")
                            remaining = verify_result.single()["remaining"]
                            
                            if remaining == 0:
                                log_graph_message("✅ BAZA DANYCH ZOSTAŁA CAŁKOWICIE WYCZYSZCZONA")
                                log_graph_message(f"📊 Usunięto {deleted_nodes} węzłów i {deleted_relations} relacji")
                                log_graph_message("🎯 Baza jest teraz kompletnie pusta i gotowa do nowych danych")
                            else:
                                log_graph_message(f"⚠️ OSTRZEŻENIE: Pozostało {remaining} węzłów w bazie")
                                
                    except Exception as e:
                        log_graph_message(f"💥 KRYTYCZNY BŁĄD podczas czyszczenia bazy: {e}")
                        log_graph_message("🆘 Operacja mogła zostać częściowo wykonana!")
                else:
                    log_graph_message("✅ Operacja czyszczenia bazy została anulowana przez użytkownika")
            else:
                log_graph_message("✅ Operacja czyszczenia bazy została anulowana")

        def run_maintenance():
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return

            def worker():
                try:
                    log_graph_message("🔄 Uruchamianie pełnej konserwacji grafu…")
                    graph_builder.run_maintenance()
                    log_graph_message("✓ Konserwacja grafu zakończona pomyślnie")
                except Exception as e:
                    log_graph_message(f"✗ Błąd konserwacji grafu: {e}")

            threading.Thread(target=worker, daemon=True).start()

        # Przyciski
        tk.Button(connection_frame, text="Testuj połączenie", 
                 command=test_connection).grid(row=0, column=4, padx=5, pady=5)
        
        operations_frame = tk.LabelFrame(graph_window, text="Operacje na grafie", 
                                        font=("Arial", 10, "bold"))
        operations_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Przyciski połączenia
        connection_buttons_frame = tk.Frame(operations_frame)
        connection_buttons_frame.pack(fill=tk.X, pady=5)
        
        connect_btn = tk.Button(connection_buttons_frame, text="Połącz z Neo4j", 
                               command=connect_to_neo4j, bg="green", fg="white", 
                               font=("Arial", 10, "bold"))
        connect_btn.pack(side=tk.LEFT, padx=5)
        
        disconnect_btn = tk.Button(connection_buttons_frame, text="Rozłącz", 
                                  command=disconnect_from_neo4j, bg="red", fg="white", 
                                  font=("Arial", 10, "bold"), state=tk.DISABLED)
        disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        # Przyciski operacji - pierwszy rząd
        operation_buttons = []
        
        graph_buttons_frame = tk.Frame(operations_frame)
        graph_buttons_frame.pack(fill=tk.X, pady=5)
        
        btn1 = tk.Button(graph_buttons_frame, text="ZAŁADUJ DANE DO BAZY", 
                        command=lambda: self.load_selected_subjects_to_neo4j(
                            [subject for subject, var in subject_vars.items() if var.get()],
                            log_graph_message, 
                            neo4j_connector
                        ),
                        state=tk.DISABLED, bg="darkgreen", fg="white", font=("Arial", 10, "bold"))
        btn1.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn1)
        
        btn2 = tk.Button(graph_buttons_frame, text="Utwórz relacje w grafie", 
                        command=create_relations, state=tk.DISABLED)
        btn2.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn2)
        
        btn3 = tk.Button(graph_buttons_frame, text="Pokaż statystyki", 
                        command=show_statistics, state=tk.DISABLED)
        btn3.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn3)
        
        btn4 = tk.Button(graph_buttons_frame, text="Konserwacja grafu", 
                        command=run_maintenance, state=tk.DISABLED,
                        bg="orange", fg="white", font=("Arial", 10, "bold"))
        btn4.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn4)
        
        # Drugi rząd przycisków - operacje usuwania
        danger_buttons_frame = tk.Frame(operations_frame)
        danger_buttons_frame.pack(fill=tk.X, pady=5)
        
        btn5 = tk.Button(danger_buttons_frame, text="USUŃ WSZYSTKIE CHUNKI", 
                        command=clear_all_chunks, state=tk.DISABLED, 
                        bg="darkred", fg="white", font=("Arial", 10, "bold"))
        btn5.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn5)
        
        # NOWY PRZYCISK - Wyczyść całą bazę
        btn6 = tk.Button(danger_buttons_frame, text="🚨 WYCZYŚĆ CAŁĄ BAZĘ 🚨", 
                        command=clear_entire_database, state=tk.DISABLED, 
                        bg="purple", fg="white", font=("Arial", 10, "bold"))
        btn6.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn6)
        
        # Zamknięcie
        def close_graph_window():
            disconnect_from_neo4j()
            graph_window.destroy()
        
        tk.Button(graph_window, text="Zamknij", command=close_graph_window,
                 bg="gray", fg="white", font=("Arial", 12, "bold"),
                 width=15, height=2).pack(pady=10)
        
        graph_window.protocol("WM_DELETE_WINDOW", close_graph_window)
        
        # Informacja startowa
        log_graph_message("=== ZARZĄDZANIE GRAFEM NEO4J ===")
        log_graph_message("1. Testuj połączenie")
        log_graph_message("2. Połącz z Neo4j")
        log_graph_message("3. Użyj przycisków operacji")
        log_graph_message("")
        log_graph_message("⚠️ UWAGA: Operacje usuwania są nieodwracalne!")

    def normalize_image_path(self, image_path):
        """
        Normalizuje ścieżkę obrazu do formatu absolutnego
        """
        try:
            path_obj = Path(image_path)
            
            # Jeśli już jest absolutna i istnieje
            if path_obj.is_absolute() and path_obj.exists():
                return str(path_obj)
            
            # POPRAWKA 1: Usuń prefix "pgverse" i wszystkie poprzedzające separatory
            path_str = str(path_obj)
            
            # Znajdź i usuń prefix "pgverse"
            if "pgverse" in path_str:
                # Znajdź indeks gdzie zaczyna się część po "pgverse"
                pgverse_index = path_str.find("pgverse")
                after_pgverse = path_str[pgverse_index + len("pgverse"):].lstrip(r'\/\\')
                path_obj = Path(after_pgverse)
            
            # POPRAWKA 2: Zbuduj ścieżkę względem base_path
            full_path = self.base_path / path_obj
            
            if full_path.exists():
                return str(full_path.resolve())
            
            # POPRAWKA 3: Jeśli nie znaleziono, spróbuj alternatywnych metod
            # Czasami ścieżka może mieć błędne separatory lub dodatkowe części
            
            # Spróbuj znaleźć plik w strukturze rezultaty
            path_parts = Path(path_obj).parts
            if len(path_parts) >= 5:  # np. ['rag_codes', 'subjects', 'TestSub', 'k1', 'rezultaty', 'wzory', 'file.png']
                try:
                    # Wyciągnij części: subject/folder/rezultaty/subfolder/filename
                    subject = path_parts[2]  # TestSub
                    folder = path_parts[3]   # k1
                    # Znajdź 'rezultaty' w ścieżce
                    rezultaty_index = -1
                    for i, part in enumerate(path_parts):
                        if part == "rezultaty":
                            rezultaty_index = i
                            break
                    
                    if rezultaty_index >= 0 and rezultaty_index < len(path_parts) - 2:
                        subfolder = path_parts[rezultaty_index + 1]  # wzory/figury/tabele
                        filename = path_parts[-1]  # file.png
                        
                        # Zbuduj nową ścieżkę
                        reconstructed_path = (self.base_path / "rag_codes" / "subjects" / 
                                            subject / folder / "rezultaty" / subfolder / filename)
                        
                        if reconstructed_path.exists():
                            return str(reconstructed_path.resolve())
                except (IndexError, ValueError):
                    pass
            
            # POPRAWKA 4: Ostatnia próba - szukaj pliku rekursywnie w folderze rezultaty
            try:
                filename = Path(path_obj).name
                # Znajdź wszystkie foldery rezultaty
                for rezultaty_folder in self.base_path.rglob("rezultaty"):
                    for found_file in rezultaty_folder.rglob(filename):
                        if found_file.exists():
                            return str(found_file.resolve())
            except Exception:
                pass
                
            return None
            
        except Exception as e:
            # DEBUG: dodaj informację o błędzie
            print(f"DEBUG normalize_image_path error for {image_path}: {e}")
            return None

    def load_data_to_neo4j(self, selected_subjects, log_function, neo4j_connector):
        """Ładuje dane do Neo4j z embeddingami obrazowymi i base64 - BEZ TWORZENIA RELACJI"""
        if not neo4j_connector:
            log_function("✗ Brak połączenia z Neo4j")
            return
            
        try:
            log_function("🔄 Inicjalizacja komponentów...")
            
            # Inicjalizacja procesorów i embeddingów
            embedder = CLIPEmbedder()
            processor = ImageTextProcessor()
            
            cuda_available = torch.cuda.is_available()
            log_function(f"🖥️ CUDA dostępne: {cuda_available}")
            
            graph_builder = TextGraphBuilder(neo4j_connector, similarity_threshold=0.8)
            log_function("✅ Komponenty zainicjalizowane pomyślnie")
            
            # Statystyki
            total_nodes_added = 0
            total_text_chunks = 0
            total_image_nodes = 0
            total_formula_nodes = 0
            total_table_nodes = 0
            total_errors = 0
            total_images_found = 0
            total_images_missing = 0
            total_base64_found = 0
            total_context_found = 0
            
            log_function("\n=== ŁADOWANIE DANYCH MULTIMODALNYCH DO NEO4J ===")
            log_function(f"📋 Przedmioty do przetworzenia: {len(selected_subjects)}")
            
            # Najpierw policz wszystkie foldery OCR
            total_ocr_folders = 0
            all_subject_folders = {}
            
            for subject_name in selected_subjects:
                log_function(f"🔍 Skanowanie przedmiotu: {subject_name}...")
                subject_path = self.subjects_path / subject_name
                
                subfolders = [item for item in subject_path.iterdir() 
                             if item.is_dir() and item.name != "__pycache__"]
                all_subject_folders[subject_name] = subfolders
                total_ocr_folders += len(subfolders)
                log_function(f"  📁 Znaleziono {len(subfolders)} folderów OCR: {[f.name for f in subfolders]}")
            
            log_function(f"\n📊 RAZEM do przetworzenia: {total_ocr_folders} folderów OCR")
            current_folder_idx = 0
            
            for subject_name in selected_subjects:
                log_function(f"\n{'='*60}")
                log_function(f"🎯 PRZETWARZANIE PRZEDMIOTU: {subject_name}")
                log_function(f"{'='*60}")
                
                subject_path = self.subjects_path / subject_name
                
                # Wczytaj konfigurację źródeł
                sources_config = {}
                sources_config_path = subject_path / "sources_config.json"
                if sources_config_path.exists():
                    try:
                        with open(sources_config_path, 'r', encoding='utf-8') as f:
                            sources_config = json.load(f)
                        log_function(f"✅ Wczytano konfigurację źródeł: {len(sources_config)} folderów")
                    except Exception as e:
                        log_function(f"⚠️ Błąd wczytywania konfiguracji źródeł: {e}")
                else:
                    log_function(f"⚠️ Brak pliku sources_config.json w {subject_name}")
                
                # Znajdź podfoldery
                subfolders = all_subject_folders[subject_name]
                log_function(f"📂 Foldery OCR w {subject_name}: {[f.name for f in subfolders]}")
                
                for subfolder in subfolders:
                    current_folder_idx += 1
                    subfolder_name = subfolder.name
                    source_type = sources_config.get(subfolder_name, "unknown")
                    
                    log_function(f"\n{'─'*50}")
                    log_function(f"📁 [{current_folder_idx}/{total_ocr_folders}] Przetwarzanie: {subject_name}/{subfolder_name}")
                    log_function(f"🏷️ Typ źródła: {source_type}")
                    log_function(f"{'─'*50}")
                    
                    detekcje_path = subfolder / "detekcje"
                    if not detekcje_path.exists():
                        log_function(f"  ⚠️ POMIJAM - Brak folderu 'detekcje' w {subfolder_name}")
                        continue
                    
                    log_function(f"✅ Folder detekcje istnieje: {detekcje_path}")
                    
                    # === KROK 1: WCZYTAJ CHUNKI TEKSTOWE Z PLIKU CHUNKS ===
                    chunks_file = detekcje_path / f"{subfolder_name}_chunks.txt"
                    text_chunks_loaded = 0
                    
                    log_function(f"\n📝 KROK 1: Ładowanie chunków tekstowych...")
                    log_function(f"🔍 Szukam pliku: {chunks_file}")
                    
                    if chunks_file.exists():
                        try:
                            log_function(f"✅ Znaleziono plik chunków: {chunks_file.name}")
                            
                            with open(chunks_file, 'r', encoding='utf-8') as f:
                                chunks_content = f.read().strip()
                            

                            # Podziel na chunki według enterów
                            text_chunks = [chunk.strip() for chunk in chunks_content.split('\n') if chunk.strip()]
                            

                            log_function(f"📊 Znaleziono {len(text_chunks)} chunków tekstowych")
                            

                            # Dodaj każdy chunk jako węzeł tekstowy
                            for chunk_idx, chunk_text in enumerate(text_chunks):
                                try:
                                    if chunk_idx % 10 == 0:  # Log co 10 chunków
                                        log_function(f"  📝 Przetwarzanie chunku {chunk_idx+1}/{len(text_chunks)}...")
                                    
                                    # Pobierz embedding tekstu
                                    text_embedding = embedder.get_text_embedding(chunk_text)
                                    
                                    if text_embedding is not None:
                                        # Utwórz UNIKALNY ID dla chunku tekstowego
                                        chunk_content = f"{subject_name}_{subfolder_name}_chunk_{chunk_idx}_{chunk_text[:50]}"
                                        chunk_id = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()
                                        unique_node_id = f"txt_{chunk_id}"
                                        
                                        # Sprawdź duplikaty
                                        with neo4j_connector.get_driver().session() as session:
                                            result = session.run(
                                                "MATCH (n {id: $node_id}) RETURN count(n) as count",
                                                node_id=unique_node_id
                                            )
                                            existing_count = result.single()["count"]
                                            
                                            if existing_count == 0:
                                                # Konwertuj ścieżkę na względną od pgverse
                                                relative_path = self.convert_to_relative_path(str(chunks_file))
                                                
                                                # Dodaj węzeł tekstowy
                                                graph_builder.insert_node(
                                                    node_id=unique_node_id,
                                                    data_type="text",
                                                    text=chunk_text,
                                                    embedding=text_embedding.tolist(),
                                                    path=relative_path,
                                                    source=source_type,
                                                    base64_data=None
                                                )
                                                
                                                total_text_chunks += 1
                                                text_chunks_loaded += 1
                                                total_nodes_added += 1
                                                
                                                if text_chunks_loaded % 20 == 0:
                                                    log_function(f"    ✅ Dodano {text_chunks_loaded} chunków tekstowych...")
                                            else:
                                                if chunk_idx % 20 == 0:  # Log duplikatów co 20
                                                    log_function(f"    ⚠️ Chunk {chunk_idx} już istnieje w bazie (pomijam)")
                                            
                                    else:
                                        log_function(f"    ⚠️ Nie udało się pobrać embeddingu dla chunku {chunk_idx}")
                                        total_errors += 1
                                        
                                except Exception as e:
                                    log_function(f"    ✗ Błąd przetwarzania chunku {chunk_idx}: {e}")
                                    total_errors += 1
                            

                            log_function(f"✅ KROK 1 ZAKOŃCZONY: Dodano {text_chunks_loaded} nowych węzłów tekstowych")
                            
                        except Exception as e:
                            log_function(f"✗ Błąd wczytywania chunków z {chunks_file.name}: {e}")
                            total_errors += 1
                    else:
                        log_function(f"⚠️ Brak pliku chunków: {chunks_file.name}")
                    
                    # === KROK 2: WCZYTAJ DANE BASE64 ===
                    log_function(f"\n📋 KROK 2: Ładowanie danych base64...")
                    base64_dict = self.load_base64_data_from_file(subfolder_name, detekcje_path)
                    log_function(f"✅ Wczytano {len(base64_dict)} zapisów base64")
                    
                    # === KROK 3: WCZYTAJ KONTEKST OBRAZÓW Z JSON ===
                    log_function(f"\n🖼️ KROK 3: Ładowanie kontekstu obrazów...")
                    context_dict = {}
                    json_files = list(detekcje_path.glob("*_filtered_context.json"))
                    log_function(f"🔍 Znaleziono {len(json_files)} plików JSON z kontekstem")
                    
                    for json_file in json_files:
                        try:
                            log_function(f"  📄 Wczytywanie: {json_file.name}")
                            
                            with open(json_file, 'r', encoding='utf-8') as f:
                                images_data = json.load(f)
                            

                            # Zbuduj słownik kontekstu: ścieżka_obrazu -> lista_tekstów_kontekstu
                            for item in images_data:
                                for image_path, context_texts in item.items():
                                    normalized_path = self.normalize_path_separators(image_path)
                                    # Połącz wszystkie teksty kontekstu w jeden string
                                    combined_context = " ".join(context_texts) if context_texts else ""
                                    context_dict[normalized_path] = combined_context
                            

                            log_function(f"  ✅ Wczytano kontekst dla {len(context_dict)} obrazów z {json_file.name}")
                            
                        except Exception as e:
                            log_function(f"  ✗ Błąd wczytywania JSON {json_file.name}: {e}")
                            total_errors += 1
                    
                    # Policz wszystkie konteksty znalezione dla tego podfolderu
                    total_context_found += len(context_dict)
                    log_function(f"✅ KROK 3 ZAKOŃCZONY: Łącznie {len(context_dict)} obrazów z kontekstem")
                    
                    # === KROK 4: DODAJ WĘZŁY OBRAZÓW/WZORÓW/TABEL ===
                    if context_dict:
                        log_function(f"\n🎨 KROK 4: Przetwarzanie {len(context_dict)} obrazów...")
                        
                        image_idx = 0
                        for image_path, context_text in context_dict.items():
                            image_idx += 1
                            try:
                                log_function(f"  🖼️ [{image_idx}/{len(context_dict)}] Sprawdzanie obrazu: {Path(image_path).name}")
                                
                                # Sprawdź czy obraz istnieje fizycznie
                                actual_image_path = self.find_actual_image_path(image_path, detekcje_path)
                                
                                if actual_image_path is None:
                                    log_function(f"    ❌ POMIJAM - Obraz nie istnieje: {Path(image_path).name}")
                                    total_images_missing += 1
                                    continue
                                
                                total_images_found += 1
                                log_function(f"    ✅ ZNALEZIONO obraz: {Path(actual_image_path).name}")
                                
                                # Określ typ danych na podstawie ścieżki
                                data_type = self.determine_data_type_from_path(image_path)
                                log_function(f"    🏷️ Typ danych: {data_type}")
                                
                                # Znajdź odpowiednie base64 na podstawie ścieżki z pgverse
                                base64_data = self.find_matching_base64(actual_image_path, base64_dict)
                                if base64_data:
                                    total_base64_found += 1
                                    log_function(f"    📋 Znaleziono dane base64")
                                else:
                                    log_function(f"    ⚠️ Brak danych base64")
                                
                                try:
                                    log_function(f"    🔄 Generowanie embeddingu obrazu...")
                                    # Pobierz embedding obrazu
                                    image_embedding = embedder.get_image_embedding(actual_image_path)
                                    
                                    if image_embedding is not None:
                                        log_function(f"    ✅ Embedding wygenerowany pomyślnie")
                                        
                                        # Utwórz UNIKALNY ID dla węzła obrazu
                                        image_content = f"{subject_name}_{subfolder_name}_{Path(actual_image_path).name}_{data_type}"
                                        image_id = hashlib.md5(image_content.encode('utf-8')).hexdigest()
                                        unique_image_node_id = f"{self.get_node_prefix(data_type)}_{image_id}"
                                        
                                        # Sprawdź duplikaty
                                        with neo4j_connector.get_driver().session() as session:
                                            result = session.run(
                                                "MATCH (n {id: $node_id}) RETURN count(n) as count",
                                                node_id=unique_image_node_id
                                            )
                                            existing_count = result.single()["count"]
                                            
                                            if existing_count == 0:
                                                # POPRAWKA 1: Przygotuj opis węzła TYLKO z kontekstem (bez prefixu typu)
                                                node_text = context_text if context_text else f"{Path(actual_image_path).name}"
                                                
                                                # Konwertuj ścieżkę na względną od pgverse
                                                relative_image_path = self.convert_to_relative_path(actual_image_path)
                                                
                                                log_function(f"    💾 Zapisywanie węzła do bazy...")
                                                # Dodaj węzeł obrazu/wzoru/tabeli z kontekstem i base64
                                                graph_builder.insert_node(
                                                    node_id=unique_image_node_id,
                                                    data_type=data_type,
                                                    text=node_text,
                                                    embedding=image_embedding.tolist(),
                                                    path=relative_image_path,
                                                    source=source_type,
                                                    base64_data=base64_data
                                                )
                                                
                                                if data_type == 'image':
                                                    total_image_nodes += 1
                                                elif data_type == 'formula':
                                                    total_formula_nodes += 1
                                                elif data_type == 'table':
                                                    total_table_nodes += 1
                                                    
                                                total_nodes_added += 1
                                                log_function(f"    ✅ Dodano węzeł {data_type}: {Path(actual_image_path).name}")
                                            else:
                                                log_function(f"    ⚠️ Węzeł {data_type} już istnieje w bazie (pomijam)")
                                    else:
                                        log_function(f"    ⚠️ Nie udało się pobrać embeddingu obrazu")
                                        
                                except Exception as e:
                                    log_function(f"    ✗ Błąd przetwarzania {data_type}: {e}")
                                    total_errors += 1
                                        
                            except Exception as e:
                                log_function(f"    ✗ Błąd przetwarzania obrazu {image_path}: {e}")
                                total_errors += 1
                    else:
                        log_function(f"⚠️ KROK 4 POMINIĘTY: Brak danych kontekstu obrazów")
                    
                    # Podsumowanie folderu OCR
                    log_function(f"\n📊 PODSUMOWANIE FOLDERU {subfolder_name}:")
                    log_function(f"  📝 Chunki tekstowe: {text_chunks_loaded}")
                    log_function(f"  🖼️ Obrazy przetworzone: {len(context_dict) if context_dict else 0}")
                    log_function(f"  📋 Base64 znalezione: {len(base64_dict)}")
            
            # === PODSUMOWANIE KOŃCOWE ===
            log_function(f"\n{'='*80}")
            log_function(f"🎉 PODSUMOWANIE ŁADOWANIA DANYCH")
            log_function(f"{'='*80}")
            log_function(f"📊 Obrazy przetwarzane: {total_images_found + total_images_missing}")
            log_function(f"✅ Obrazy znalezione: {total_images_found}")
            log_function(f"❌ Obrazy pominięte (nie istnieją): {total_images_missing}")
            log_function(f"📋 Dane base64 znalezione: {total_base64_found}")
            log_function(f"📝 Kontekst znaleziony dla: {total_context_found} obrazów")
            log_function(f"")
            log_function(f"=== WĘZŁY DODANE DO BAZY ===")
            log_function(f"🖼️ Węzłów obrazów: {total_image_nodes}")
            log_function(f"🧮 Węzłów wzorów: {total_formula_nodes}")
            log_function(f"📊 Węzłów tabel: {total_table_nodes}")
            log_function(f"📝 Węzłów tekstowych: {total_text_chunks}")
            log_function(f"🎯 RAZEM węzłów: {total_nodes_added}")
            log_function(f"⚠️ Błędy: {total_errors}")
            
            # NOWE: Informacja o braku tworzenia relacji
            log_function(f"\n📋 UWAGA: Węzły zostały dodane BEZ relacji.")
            log_function(f"🔗 Aby utworzyć relacje podobieństwa między węzłami:")
            log_function(f"   1. Użyj przycisku 'Utwórz relacje tekstowe'")
            log_function(f"   2. Lub użyj przycisku 'Konserwacja grafu' (pełna konserwacja)")
            
            if total_nodes_added > 0:
                log_function("\n🎉 ŁADOWANIE WĘZŁÓW ZAKOŃCZONE POMYŚLNIE!")
                log_function(f"📈 Graf zawiera teraz:")
                log_function(f"  • {total_text_chunks} węzłów tekstowych z chunków")
                log_function(f"  • {total_image_nodes} węzłów obrazów z kontekstem")
                log_function(f"  • {total_formula_nodes} węzłów wzorów z kontekstem")
                log_function(f"  • {total_table_nodes} węzłów tabel z kontekstem")
                log_function(f"  • {total_base64_found} węzłów z danymi base64")
                log_function(f"  • BRAK relacji - dodaj je osobno!")
            else:
                log_function("⚠️ Nie dodano żadnych danych - sprawdź logi błędów")
                
        except Exception as e:
            log_function(f"✗ Krytyczny błąd ładowania danych: {e}")
            import traceback
            log_function(f"Traceback: {traceback.format_exc()}")

    def load_base64_data_from_file(self, ocr_folder_name, detekcje_path):
        """
        Ładuje dane base64 z pliku {OCRfolder}_base64.txt
        Format: <image/path/base64_data> przeplatane z chunkami tekstu
        Zwraca słownik: {ścieżka_obrazu: base64_string}
        """
        base64_file = detekcje_path / f"{ocr_folder_name}_base64.txt"
        base64_dict = {}
        
        print(f"DEBUG load_base64_data_from_file:")
        print(f"  Szukam pliku: {base64_file}")
        print(f"  Plik istnieje: {base64_file.exists()}")
        
        if not base64_file.exists():
            print(f"  ❌ Plik base64 nie istnieje!")
            return base64_dict
            
        try:
            with open(base64_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"  📄 Rozmiar pliku: {len(content)} znaków")
            
            # NOWY PATTERN: <image/ścieżka/base64_data> gdzie base64_data jest w środku
            import re
            
            # Pattern: <image/ następnie ścieżka / następnie base64_data >
            # Grupa 1: ścieżka (wszystko między <image/ a ostatnim /)
            # Grupa 2: base64_data (wszystko między ostatnim / a >)
            pattern = r'<image/(.+?)/([A-Za-z0-9+/=]+)>'
            matches = list(re.finditer(pattern, content, re.DOTALL))
            
            print(f"  🔍 Znaleziono {len(matches)} wzorców obrazów")
            
            for i, match in enumerate(matches):
                full_match = match.group(0)
                image_path = match.group(1)  # Ścieżka obrazu
                base64_data = match.group(2)  # Dane base64
                
                print(f"  🔍 [{i+1}] Znaleziono obraz:")
                print(f"    Pełny wzorzec: {full_match[:100]}...")
                print(f"    Ścieżka: {image_path}")
                print(f"    Base64 długość: {len(base64_data)} znaków")
                print(f"    Base64 początek: {base64_data[:50]}...")
                print(f"    Base64 koniec: ...{base64_data[-50:]}")
                
                # Normalizuj ścieżkę (zamień backslashe na forward slashe)
                normalized_path = image_path.replace('\\', '/')
                
                # Dodaj brakujące znaki '=' na końcu base64 jeśli potrzebne
                # Base64 musi mieć długość podzielną przez 4
                padding_needed = len(base64_data) % 4
                if padding_needed:
                    base64_data += '=' * (4 - padding_needed)
                
                base64_dict[normalized_path] = base64_data
                print(f"    ✅ DODANO base64 dla: {normalized_path}")
                print(f"      Finalna długość base64: {len(base64_data)} znaków")
            
            print(f"  📊 PODSUMOWANIE: Wczytano {len(base64_dict)} obrazów z base64")
            
            # DEBUG: Pokaż wszystkie klucze w słowniku
            if base64_dict:
                print(f"  📋 Klucze w base64_dict:")
                for key in base64_dict.keys():
                    print(f"    - {key}")
        
        except Exception as e:
            print(f"  ❌ Błąd czytania pliku: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")

        return base64_dict

    def find_matching_base64(self, actual_image_path, base64_dict):
        """
        Znajduje odpowiednie dane base64 dla ścieżki obrazu
        Porównuje końcowe części ścieżek (np. wzory\filename.png)
        """
        try:
            filename = Path(actual_image_path).name
            
            print(f"DEBUG find_matching_base64:")
            print(f"  actual_image_path: {actual_image_path}")
            print(f"  filename: {filename}")
            
            # NOWE: Wyciągnij końcową część ścieżki (folder\filename)
            # np. C:\...\detekcje\wzory\k1_page1_formula1.png -> wzory\k1_page1_formula1.png
            path_obj = Path(actual_image_path)
            if len(path_obj.parts) >= 2:
                # Weź ostatnie 2 części: folder\filename
                ending_part = path_obj.parts[-2] + '\\' + path_obj.parts[-1]
            else:
                ending_part = filename
            
            print(f"  ending_part do porównania: {ending_part}")
            
            # GŁÓWNE WYSZUKIWANIE: Sprawdź czy jakaś ścieżka w base64_dict kończy się na ending_part
            for stored_path, base64_data in base64_dict.items():
                # Normalizuj stored_path do backslashes dla porównania
                normalized_stored = stored_path.replace('/', '\\')
                
                print(f"  🔍 Sprawdzam stored_path: {stored_path}")
                print(f"    normalized_stored: {normalized_stored}")
                print(f"    czy kończy się na '{ending_part}': {normalized_stored.endswith(ending_part)}")
                
                if normalized_stored.endswith(ending_part):
                    print(f"  ✅ ZNALEZIONO dopasowanie!")
                    print(f"    stored_path: {stored_path}")
                    print(f"    ending_part: {ending_part}")
                    return base64_data
            
            # Fallback 1: Sprawdź tylko po nazwie pliku
            print(f"  🔍 Fallback 1 - szukam po nazwie pliku: {filename}")
            for stored_path, base64_data in base64_dict.items():
                if stored_path.endswith(filename):
                    print(f"  ✅ ZNALEZIONO po nazwie pliku: {filename}")
                   
                    return base64_data
            
            # Fallback 2: Sprawdź z forward slashes
            ending_part_forward = ending_part.replace('\\', '/')
            print(f"  🔍 Fallback 2 - szukam z forward slashes: {ending_part_forward}")
            for stored_path, base64_data in base64_dict.items():
                if stored_path.endswith(ending_part_forward):
                    print(f"  ✅ ZNALEZIONO z forward slashes!")
                    return base64_data
            
            print(f"  ❌ NIE ZNALEZIONO dopasowania dla: {ending_part}")
            return None
            
        except Exception as e:
            print(f"DEBUG find_matching_base64 ERROR: {e}")
            return None

    def extract_path_from_pgverse(self, path):
        """
        Wyciąga część ścieżki zaczynającą się od folderu po 'pgverse'
        Np. pgverse/rag_codes/subjects/... -> rag_codes/subjects/...
        """
        try:
            if 'pgverse' in path:
                parts = path.split('/')
                pgverse_index = -1
                for i, part in enumerate(parts):
                    if part.lower() == 'pgverse':
                        pgverse_index = i
                        break
            
                if pgverse_index >= 0 and pgverse_index < len(parts) - 1:
                    return '/'.join(parts[pgverse_index + 1:])
        
            return path
        except Exception:
            return path

    # Dodaj brakujące funkcje pomocnicze
    def determine_data_type_from_path(self, image_path):
        """Określa typ danych na podstawie ścieżki"""
        path_lower = image_path.lower()
        
        for folder_name, data_type in self.folder_type_mapping.items():
            if folder_name in path_lower:
                return data_type
        
        # Domyślnie image
        return "image"
    
    def get_node_prefix(self, data_type):
        """Zwraca prefix dla ID węzła na podstawie typu danych"""
        if data_type == "image":
            return "img"
        elif data_type == "formula":
            return "frm"
        elif data_type == "table":
            return "tbl"
        else:
            return "unk"
    
    def find_actual_image_path(self, image_path, detekcje_path):
        """Znajduje rzeczywistą ścieżkę do obrazu"""
        try:
            # Usuń prefix "pgverse" jeśli istnieje
            normalized_path = self.normalize_path_separators(image_path)
            
            # Jeśli ścieżka zawiera "pgverse", znajdź część po nim
            if "pgverse" in normalized_path:
                parts = normalized_path.split("pgverse/")
                if len(parts) > 1:
                    relative_path = parts[1]
                    full_path = self.base_path / relative_path
                    if full_path.exists():
                        return str(full_path)
            
            # Spróbuj znaleźć obraz względem detekcje_path
            filename = Path(image_path).name
            
            # Sprawdź w podfolderach detekcje (figury, wzory, tabele)
            for subfolder in detekcje_path.iterdir():
                if subfolder.is_dir():
                    potential_file = subfolder / filename
                    if potential_file.exists():
                        return str(potential_file)
            
            # Sprawdź bezpośrednio w detekcje
            potential_file = detekcje_path / filename
            if potential_file.exists():
                return str(potential_file)
            
            return None
            
        except Exception as e:
            print(f"DEBUG find_actual_image_path error for {image_path}: {e}")
            return None

    def load_selected_subjects_to_neo4j(self, selected_subjects_for_loading, log_function, neo4j_connector):
        """Ładuje wybrane przedmioty do Neo4j z walidacją - uruchamia w osobnym wątku"""
        if not selected_subjects_for_loading:
            log_function("⚠️ Nie wybrano żadnego przedmiotu do załadowania!")
            log_function("📋 Zaznacz przynajmniej jeden przedmiot w sekcji 'Wybór przedmiotów do załadowania'")
            messagebox.showwarning("Brak wyboru przedmiotów", 
                                  "Nie wybrano żadnego przedmiotu do załadowania!\n\n"
                                  "Zaznacz przynajmniej jeden przedmiot w sekcji\n"
                                  "'Wybór przedmiotów do załadowania' przed\n"
                                  "kliknięciem przycisku 'ZAŁADUJ DANE DO BAZY'.")
            return
            
        log_function(f"🎯 Rozpoczynam ładowanie wybranych przedmiotów: {', '.join(selected_subjects_for_loading)}")
        log_function("🔄 Uruchamiam ładowanie w osobnym wątku...")
        
        # Uruchom ładowanie w osobnym wątku, żeby nie zawiesić GUI
        def load_in_thread():
            try:
                self.load_data_to_neo4j(selected_subjects_for_loading, log_function, neo4j_connector)
                log_function("🎉 ŁADOWANIE ZAKOŃCZONE - operacja wykonana w tle")
            except Exception as e:
                log_function(f"💥 BŁĄD KRYTYCZNY podczas ładowania: {e}")
                import traceback
                log_function(f"Traceback: {traceback.format_exc()}")
        
        loading_thread = threading.Thread(target=load_in_thread)
        loading_thread.daemon = True
        loading_thread.start()
        
        log_function("✅ Wątek ładowania uruchomiony - sprawdzaj logi poniżej...")

    def run(self):
        """Uruchamia aplikację"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SubjectSelectorApp()
    app.run()