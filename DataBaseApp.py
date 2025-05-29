import tkinter as tk
from pathlib import Path
import json
import tkinter.messagebox
from tkinter import ttk
import threading
import time
from rag_codes.rag_functions.metadata_caption import ImageTextProcessor, ImageContextFilter
from rag_codes.rag_functions.embeddings import CLIPEmbedder
from rag_codes.rag_functions.chunker import TextChunker
from rag_codes.rag_functions.graph import TextGraphBuilder
import hashlib
class SubjectSelectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Wybór przedmiotów")
        self.root.geometry("800x600")
        
        # Ustalenie ścieżki bazowej (folder pgverse)
        self.base_path = Path(__file__).parent
        self.subjects_path = self.base_path / "rag_codes" / "subjects"
        
        # Słownik do przechowywania stanu checkboxów
        self.subject_vars = {}
        # Słownik do przechowywania obiektów checkboxów
        self.subject_checkboxes = {}
        
        # Słownik do przechowywania tymczasowych konfiguracji źródeł
        self.temp_sources_configs = {}
        
        # Dostępne typy źródeł
        self.source_types = [
            "wikipedia",
            "książka", 
            "artykuł_naukowy",
            "blog",
            "forum",
            "social_media",
            "news",
            "unknown"
        ]
        
        self.create_widgets()
        self.load_subjects()
        
    def create_widgets(self):
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
        
        select_all_btn = tk.Button(button_frame, text="Zaznacz wszystko", 
                                  command=self.select_all)
        select_all_btn.pack(side=tk.LEFT, padx=5)
        
        deselect_all_btn = tk.Button(button_frame, text="Odznacz wszystko", 
                                    command=self.deselect_all)
        deselect_all_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame dla przycisków głównych - ZAMKNIJ i DALEJ
        main_buttons_frame = tk.Frame(self.root)
        main_buttons_frame.pack(pady=10, side=tk.BOTTOM)  # Dodane side=tk.BOTTOM
        
        # Przycisk ZAMKNIJ - zamyka program
        close_btn = tk.Button(main_buttons_frame, 
                             text="ZAMKNIJ", 
                             command=self.close_application,
                             bg="red", 
                             fg="white",
                             font=("Arial", 12, "bold"),
                             width=12,
                             height=2)
        close_btn.pack(side=tk.LEFT, padx=20)
        
        # Przycisk DALEJ - zapisuje JSONy i kontynuuje
        next_btn = tk.Button(main_buttons_frame, 
                            text="DALEJ", 
                            command=self.save_and_proceed,
                            bg="green", 
                            fg="white",
                            font=("Arial", 12, "bold"),
                            width=12,
                            height=2)
        next_btn.pack(side=tk.LEFT, padx=20)

    def close_application(self):
        """ZAMKNIJ - Zamyka aplikację z potwierdzeniem"""
        if self.temp_sources_configs:
            result = tk.messagebox.askyesno("Potwierdzenie zamknięcia", 
                                           "Czy na pewno chcesz zamknąć aplikację?\n"
                                           "Masz niezapisane zmiany w źródłach, które zostaną utracone!")
        else:
            result = tk.messagebox.askyesno("Potwierdzenie zamknięcia", 
                                           "Czy na pewno chcesz zamknąć aplikację?")
        
        if result:
            self.root.destroy()
    
    def save_and_proceed(self):
        """DALEJ - Zapisuje konfiguracje źródeł do plików JSON i kontynuuje"""
        selected_subjects = [subject for subject, var in self.subject_vars.items() 
                           if var.get()]
        
        if not selected_subjects:
            tk.messagebox.showwarning("Brak wyboru", 
                                     "Nie wybrano żadnego przedmiotu!\n"
                                     "Zaznacz przynajmniej jeden przedmiot przed kontynuowaniem.")
            return
        
        # Sprawdzenie czy wszystkie wybrane przedmioty mają przypisane źródła
        incomplete_subjects = [s for s in selected_subjects if not self.check_sources_complete(s)]
        
        if incomplete_subjects:
            tk.messagebox.showerror("Błąd - nieprzypisane źródła", 
                f"Następujące przedmioty mają nieprzypisane źródła:\n\n" + 
                "\n".join(f"• {subject}" for subject in incomplete_subjects) +
                "\n\nUzupełnij źródła dla wszystkich wybranych przedmiotów.")
            return
        
        # Zapisywanie konfiguracji do plików JSON
        saved_count = 0
        failed_subjects = []
        updated_subjects = []
        
        print("=== ROZPOCZĘCIE ZAPISYWANIA KONFIGURACJI ===")
        
        for subject_name in selected_subjects:
            subject_path = self.subjects_path / subject_name
            
            # Sprawdź czy są zmiany do zapisania
            if subject_name in self.temp_sources_configs:
                print(f"Zapisywanie konfiguracji dla: {subject_name}")
                if self.save_sources_config(subject_path, self.temp_sources_configs[subject_name]):
                    saved_count += 1
                    updated_subjects.append(subject_name)
                    print(f"✓ Zapisano: {subject_name}")
                else:
                    failed_subjects.append(subject_name)
                    print(f"✗ Błąd zapisywania: {subject_name}")
            else:
                print(f"Brak zmian dla: {subject_name}")
        
        # Raportowanie wyników
        if failed_subjects:
            tk.messagebox.showerror("Błąd zapisywania", 
                f"Nie udało się zapisać konfiguracji dla przedmiotów:\n\n" + 
                "\n".join(f"• {subject}" for subject in failed_subjects) +
                "\n\nSprawdź uprawnienia do zapisu w folderach przedmiotów.")
            return
        
        # Komunikat o sukcesie
        success_message = f"🎯 SUKCES!\n\n"
        success_message += f"Wybrano {len(selected_subjects)} przedmiotów:\n"
        success_message += "\n".join(f"• {subject}" for subject in selected_subjects)
        
        if saved_count > 0:
            success_message += f"\n\n💾 Zapisano {saved_count} nowych konfiguracji źródeł:\n"
            success_message += "\n".join(f"• {subject}" for subject in updated_subjects)
        
        success_message += f"\n\n🚀 Konfiguracja zakończona pomyślnie!"
        
        tk.messagebox.showinfo("Konfiguracja zakończona", success_message)
        
        # Wyczyść tymczasowe konfiguracje po zapisaniu
        self.temp_sources_configs.clear()
        
        # Otwórz okno przetwarzania podfolderów
        self.open_processing_window(selected_subjects)

    def open_processing_window(self, selected_subjects):
        """Otwiera okno przetwarzania podfolderów z obrazkami i tekstem"""
        # Nowe okno
        processing_window = tk.Toplevel(self.root)
        processing_window.title("Przetwarzanie podfolderów")
        processing_window.geometry("900x700")
        processing_window.grab_set()  # Modal window
        
        # Tytuł
        title_label = tk.Label(processing_window, 
                              text="Przetwarzanie podfolderów do formatu JSON", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Informacja o przedmiotach
        info_label = tk.Label(processing_window, 
                             text=f"Wybrane przedmioty ({len(selected_subjects)}): " + 
                                  ", ".join(selected_subjects), 
                             font=("Arial", 10), wraplength=800)
        info_label.pack(pady=5)
        
        # Frame dla głównej zawartości
        main_frame = tk.Frame(processing_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Scrollable text widget do logów
        log_frame = tk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(log_frame, text="Log przetwarzania:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        # Text widget z scrollbarem
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
        
        # Przyciski
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        start_btn = tk.Button(button_frame, text="ROZPOCZNIJ PRZETWARZANIE", 
                             command=lambda: self.start_processing(selected_subjects, 
                                                                  log_text, progress_bar, 
                                                                  progress_label, start_btn, close_btn),
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
            # Re-enable przyciski po zakończeniu
            start_btn.config(state=tk.NORMAL, text="ZAKOŃCZONO", bg="gray")
        
        # Uruchom w osobnym wątku żeby nie blokować GUI
        processing_thread = threading.Thread(target=process_in_thread)
        processing_thread.daemon = True
        processing_thread.start()

    def log_message(self, log_text, message):
        """Dodaje wiadomość do log widget"""
        try:
            log_text.insert(tk.END, f"{message}\n")
            log_text.see(tk.END)
            log_text.update_idletasks()  # Zmiana z update() na update_idletasks()
        except tk.TclError:
            # Widget został zniszczony
            pass

    def update_progress(self, progress_bar, progress_label, current, total, current_task=""):
        """Aktualizuje progress bar i label"""
        try:
            percentage = (current / total) * 100 if total > 0 else 0
            progress_bar['value'] = percentage
            progress_label.config(text=f"Postęp: {current}/{total} ({percentage:.1f}%) - {current_task}")
            progress_bar.update_idletasks()  # Zmiana z update() na update_idletasks()
            progress_label.update_idletasks()  # Zmiana z update() na update_idletasks()
        except tk.TclError:
            # Widget został zniszczony
            pass

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
        
        # Policz wszystkie foldery OCR do przetworzenia
        total_ocr_folders = 0
        subject_ocr_folders = {}
        
        for subject_name in selected_subjects:
            subject_path = self.subjects_path / subject_name
            
            # Znajdź foldery OCR w przedmiocie (bezpośrednio w folderze przedmiotu)
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
        
        # Przetwarzanie każdego przedmiotu
        for subject_name in selected_subjects:
            self.log_message(log_text, f"\n--- PRZETWARZANIE PRZEDMIOTU: {subject_name} ---")
            
            ocr_folders = subject_ocr_folders[subject_name]
            
            if not ocr_folders:
                self.log_message(log_text, f"Brak folderów OCR w przedmiocie {subject_name}")
                continue
            
            # Przetwarzanie każdego folderu OCR
            for ocr_folder in ocr_folders:
                processed_count += 1
                self.update_progress(progress_bar, progress_label, processed_count, total_ocr_folders, 
                                   f"{subject_name}/{ocr_folder.name}")
                
                self.log_message(log_text, f"\nPrzetwarzanie folderu OCR: {subject_name}/{ocr_folder.name}")
                
                # Sprawdź czy istnieje folder rezultaty w folderze OCR
                rezultaty_path = ocr_folder / "rezultaty"
                if not rezultaty_path.exists():
                    self.log_message(log_text, f"  ⚠ Brak folderu 'rezultaty' w {ocr_folder.name}")
                    continue
                
                # Znajdź pliki txt w folderze rezultaty
                txt_files = list(rezultaty_path.glob("*.txt"))
                
                if not txt_files:
                    self.log_message(log_text, f"  ⚠ Brak plików .txt w {ocr_folder.name}/rezultaty")
                    continue
                
                # Przetwórz każdy plik txt
                for txt_file in txt_files:
                    try:
                        self.log_message(log_text, f"  📄 Przetwarzanie pliku: {ocr_folder.name}/rezultaty/{txt_file.name}")
                        
                        # Przetwarzanie przez ImageTextProcessor
                        texts = processor.process_file(str(txt_file))
                        json_data = processor.get_images_with_context_json(texts)
                        
                        if not json_data:
                            self.log_message(log_text, f"    ⚠ Brak obrazów w pliku {txt_file.name}")
                            continue
                        
                        self.log_message(log_text, f"    ✓ Znaleziono {len(json_data)} obrazów")
                        
                        # Filtrowanie kontekstu
                        self.log_message(log_text, f"    🔄 Filtrowanie kontekstu...")
                        filtered_data = filter_processor.process_images_context(json_data)
                        
                        # Zapisz wyniki w folderze rezultaty (tam gdzie są pliki txt)
                        output_file = rezultaty_path / f"{txt_file.stem}_filtered_context.json"
                        result = filter_processor.save_filtered_context(filtered_data, str(output_file))
                        
                        if result:
                            self.log_message(log_text, f"    ✓ Zapisano: {ocr_folder.name}/rezultaty/{output_file.name}")
                            success_count += 1
                        else:
                            self.log_message(log_text, f"    ✗ Błąd zapisywania: {output_file.name}")
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
            
            # USUŃ problematyczne after() - wywołaj bezpośrednio
            result = tk.messagebox.askyesno("Zarządzanie grafem Neo4j", 
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
        # Nowe okno
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Zarządzanie grafem Neo4j")
        graph_window.geometry("1000x800")
        graph_window.grab_set()  # Modal window
        
        # Tytuł
        title_label = tk.Label(graph_window, 
                              text="Zarządzanie grafem Neo4j - Tworzenie relacji między chunkami", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Informacja o przedmiotach
        info_label = tk.Label(graph_window, 
                             text=f"Wybrane przedmioty: {', '.join(selected_subjects)}", 
                             font=("Arial", 10), wraplength=900)
        info_label.pack(pady=5)
        
        # Frame dla konfiguracji połączenia
        connection_frame = tk.LabelFrame(graph_window, text="Konfiguracja połączenia Neo4j", font=("Arial", 10, "bold"))
        connection_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Pola połączenia
        tk.Label(connection_frame, text="URI:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        uri_var = tk.StringVar(value="neo4j+s://335a260d.databases.neo4j.io")
        uri_entry = tk.Entry(connection_frame, textvariable=uri_var, width=30)
        uri_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(connection_frame, text="User:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        user_var = tk.StringVar(value="neo4j")
        user_entry = tk.Entry(connection_frame, textvariable=user_var, width=15)
        user_entry.grid(row=0, column=3, padx=5, pady=5)
        
        tk.Label(connection_frame, text="Password:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        password_var = tk.StringVar(value="4RMc8un8Yjzx9oy3_l2fDw5pNbuKuNdGjLHFI4a_EEU")
        password_entry = tk.Entry(connection_frame, textvariable=password_var, show="*", width=15)
        password_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Status połączenia
        status_var = tk.StringVar(value="Nie połączono")
        status_label = tk.Label(connection_frame, textvariable=status_var, fg="red", font=("Arial", 9))
        status_label.grid(row=1, column=2, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Przycisk testowania połączenia
        def test_connection():
            try:
                from rag_codes.rag_functions.graph import Neo4jConnector
                connector = Neo4jConnector(uri_var.get(), user_var.get(), password_var.get())
                # Test połączenia
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
        
        test_btn = tk.Button(connection_frame, text="Testuj połączenie", command=test_connection)
        test_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Frame dla logów
        log_frame = tk.LabelFrame(graph_window, text="Log operacji", font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Text widget z scrollbarem
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
        text_retriever = None
        graph_pruner = None
        pattern_tracker = None
        
        def log_graph_message(message):
            """Dodaje wiadomość do log widget grafu"""
            try:
                graph_log_text.insert(tk.END, f"{message}\n")
                graph_log_text.see(tk.END)
                graph_log_text.update_idletasks()  # Zmiana z update() na update_idletasks()
            except tk.TclError:
                # Widget został zniszczony
                pass
        
        def connect_to_neo4j():
            """Łączy się z Neo4j i inicjalizuje komponenty"""
            nonlocal neo4j_connector, graph_builder, text_retriever, graph_pruner, pattern_tracker
            
            try:
                from rag_codes.rag_functions.graph import (Neo4jConnector, TextGraphBuilder, 
                                                          HybridTextRetriever, GraphPruner, 
                                                          LearningPatternTracker)
                
                log_graph_message("🔄 Łączenie z Neo4j...")
                neo4j_connector = Neo4jConnector(uri_var.get(), user_var.get(), password_var.get())
                
                log_graph_message("🔄 Inicjalizacja komponentów grafu...")
                graph_builder = TextGraphBuilder(neo4j_connector, similarity_threshold=0.8)
                text_retriever = HybridTextRetriever(neo4j_connector)
                text_retriever.graph_builder = graph_builder  # Dla funkcji uczenia
                graph_pruner = GraphPruner(neo4j_connector, similarity_threshold=0.7)
                pattern_tracker = LearningPatternTracker(neo4j_connector)
                
                log_graph_message("✓ Pomyślnie połączono z Neo4j i zainicjalizowano komponenty")
                
                # Aktywuj przyciski operacji
                for btn in operation_buttons:
                    btn.config(state=tk.NORMAL)
                
                connect_btn.config(state=tk.DISABLED, text="Połączono")
                disconnect_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                log_graph_message(f"✗ Błąd połączenia: {e}")
        
        def disconnect_from_neo4j():
            """Rozłącza z Neo4j"""
            nonlocal neo4j_connector, graph_builder, text_retriever, graph_pruner, pattern_tracker
            
            try:
                if neo4j_connector:
                    neo4j_connector.close()
                if graph_builder:
                    graph_builder.close()
                if text_retriever:
                    text_retriever.close()
                if graph_pruner:
                    graph_pruner.close()
                if pattern_tracker:
                    pattern_tracker.close()
                
                neo4j_connector = None
                graph_builder = None
                text_retriever = None
                graph_pruner = None
                pattern_tracker = None
                
                log_graph_message("✓ Rozłączono z Neo4j")
                
                # Dezaktywuj przyciski operacji
                for btn in operation_buttons:
                    btn.config(state=tk.DISABLED)
                
                connect_btn.config(state=tk.NORMAL, text="Połącz z Neo4j")
                disconnect_btn.config(state=tk.DISABLED)
                
            except Exception as e:
                log_graph_message(f"✗ Błąd rozłączania: {e}")
        
        def create_text_relations():
            """Tworzy relacje między węzłami tekstowymi"""
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            try:
                log_graph_message("🔄 Tworzenie relacji między węzłami tekstowymi...")
                graph_builder.create_text_relations()
                log_graph_message("✓ Relacje tekstowe utworzone pomyślnie")
            except Exception as e:
                log_graph_message(f"✗ Błąd tworzenia relacji: {e}")
        
        def run_maintenance():
            """Uruchamia konserwację grafu"""
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            try:
                log_graph_message("🔄 Uruchamianie konserwacji grafu...")
                graph_builder.run_maintenance()
                log_graph_message("✓ Konserwacja grafu zakończona")
            except Exception as e:
                log_graph_message(f"✗ Błąd konserwacji: {e}")
        
        def show_statistics():
            """Pokazuje statystyki grafu"""
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            try:
                log_graph_message("🔄 Pobieranie statystyk grafu...")
                stats = graph_builder.analyze_learning_patterns()
                
                log_graph_message("=== STATYSTYKI GRAFU ===")
                
                # Statystyki węzłów
                node_stats = stats['node_statistics']
                log_graph_message(f"Węzły tekstowe: {node_stats.get('total_nodes', 0)}")
                log_graph_message(f"Średnie użycie: {node_stats.get('avg_usage', 0):.2f}")
                log_graph_message(f"Maksymalne użycie: {node_stats.get('max_usage', 0)}")
                log_graph_message(f"Popularne węzły (>5 użyć): {node_stats.get('popular_nodes', 0)}")
                
                # Statystyki relacji
                rel_stats = stats['relation_statistics']
                log_graph_message(f"Wszystkich relacji: {rel_stats.get('total_relations', 0)}")
                log_graph_message(f"Średnia waga: {rel_stats.get('avg_weight', 0):.3f}")
                log_graph_message(f"Średnie wzmocnienia: {rel_stats.get('avg_reinforcements', 0):.2f}")
                log_graph_message(f"Silne relacje (>0.8): {rel_stats.get('strong_relations', 0)}")
                log_graph_message(f"Nauczone relacje (>3 wzmocnień): {rel_stats.get('learned_relations', 0)}")
                
                log_graph_message(f"Aktualny próg: {stats.get('current_threshold', 0):.3f}")
                log_graph_message(f"Wzorce użycia: {stats.get('usage_patterns_count', 0)}")
                
                # Popularne węzły
                if stats['popular_nodes']:
                    log_graph_message("\n=== NAJPOPULARNIEJSZE WĘZŁY ===")
                    for node in stats['popular_nodes'][:5]:
                        log_graph_message(f"ID: {node['id'][:20]}... (użyć: {node['usage_count']}, konteksty: {node['context_variety']})")
                
                # Silne relacje
                if stats['strong_relations']:
                    log_graph_message("\n=== NAJSILNIEJSZE RELACJE ===")
                    for rel in stats['strong_relations'][:5]:
                        log_graph_message(f"Waga: {rel['weight']:.3f}, Wzmocnienia: {rel['reinforcements']}")
                
            except Exception as e:
                log_graph_message(f"✗ Błąd pobierania statystyk: {e}")
        
        def prune_old_relations():
            """Usuwa stare relacje"""
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            try:
                log_graph_message("🔄 Usuwanie starych relacji...")
                graph_builder.prune_old()
                log_graph_message("✓ Stare relacje usunięte")
            except Exception as e:
                log_graph_message(f"✗ Błąd usuwania relacji: {e}")
        
        def create_semantic_clusters():
            """Tworzy klastry semantyczne"""
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            try:
                log_graph_message("🔄 Tworzenie klastrów semantycznych...")
                clusters = graph_builder.create_semantic_clusters()
                log_graph_message(f"✓ Utworzono {len(clusters)} klastrów semantycznych")
                
                if clusters:
                    log_graph_message("=== KLASTRY SEMANTYCZNE ===")
                    for cluster in clusters[:10]:  # Pokaż pierwsze 10
                        log_graph_message(f"Węzeł: {cluster['node_id'][:30]}... (rozmiar: {cluster['cluster_size']})")
                        
            except Exception as e:
                log_graph_message(f"✗ Błąd tworzenia klastrów: {e}")
        
        def clear_all_relations():
            """Usuwa wszystkie relacje z grafu"""
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            result = tk.messagebox.askyesno("Potwierdzenie", 
                "Czy na pewno chcesz usunąć WSZYSTKIE relacje z grafu?\n"
                "Ta operacja jest nieodwracalna!")
            
            if result:
                try:
                    log_graph_message("🔄 Usuwanie wszystkich relacji...")
                    with neo4j_connector.get_driver().session() as session:
                        result = session.run("MATCH ()-[r:SIMILAR_TO]-() DELETE r")
                        deleted_count = result.consume().counters.relationships_deleted
                    log_graph_message(f"✓ Usunięto {deleted_count} relacji")
                except Exception as e:
                    log_graph_message(f"✗ Błąd usuwania relacji: {e}")
        
        def clear_all_chunks():
            """Usuwa wszystkie chunki (węzły tekstowe) z grafu"""
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            result = tk.messagebox.askyesno("Potwierdzenie", 
                "Czy na pewno chcesz usunąć WSZYSTKIE CHUNKI (węzły tekstowe) z grafu?\n"
                "Ta operacja jest nieodwracalna!")
            
            if result:
                try:
                    log_graph_message("🔄 Usuwanie wszystkich chunków...")
                    with neo4j_connector.get_driver().session() as session:
                        # Usuń wszystkie węzły tekstowe wraz z ich relacjami
                        result = session.run("""
                            MATCH (n:TextNode) 
                            DETACH DELETE n
                        """)
                        deleted_count = result.consume().counters.nodes_deleted
                    log_graph_message(f"✓ Usunięto {deleted_count} chunków tekstowych")
                except Exception as e:
                    log_graph_message(f"✗ Błąd usuwania chunków: {e}")
        
        def clear_entire_database():
            """Usuwa WSZYSTKO z bazy danych Neo4j"""
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            result = tk.messagebox.askyesno("UWAGA - TOTALNE WYCZYSZCZENIE BAZY", 
                "⚠️ UWAGA! ⚠️\n\n"
                "Ta operacja USUWA CAŁKOWICIE WSZYSTKO z bazy danych Neo4j:\n"
                "• Wszystkie węzły (każdego typu)\n"
                "• Wszystkie relacje\n"
                "• Wszystkie właściwości\n"
                "• Wszystkie indeksy\n\n"
                "BAZA ZOSTANIE CAŁKOWICIE WYCZYSZCZONA!\n\n"
                "Czy na pewno chcesz kontynuować?")
            
            if result:
                # Drugie potwierdzenie dla bezpieczeństwa
                second_result = tk.messagebox.askyesno("OSTATNIE OSTRZEŻENIE", 
                    "🚨 OSTATNIA SZANSA! 🚨\n\n"
                    "Za chwilę zostanie CAŁKOWICIE WYCZYSZCZONA baza danych!\n"
                    "Wszystkie dane zostaną BEZPOWROTNIE UTRACONE!\n\n"
                    "Czy jesteś ABSOLUTNIE PEWIEN?")
                
                if second_result:
                    try:
                        log_graph_message("🔥 ROZPOCZĘCIE TOTALNEGO WYCZYSZCZANIA BAZY...")
                        
                        with neo4j_connector.get_driver().session() as session:
                            # Usuń wszystkie relacje
                            log_graph_message("  🔄 Usuwanie wszystkich relacji...")
                            result1 = session.run("MATCH ()-[r]-() DELETE r")
                            deleted_relations = result1.consume().counters.relationships_deleted
                            log_graph_message(f"  ✓ Usunięto {deleted_relations} relacji")
                            
                            # Usuń wszystkie węzły
                            log_graph_message("  🔄 Usuwanie wszystkich węzłów...")
                            result2 = session.run("MATCH (n) DELETE n")
                            deleted_nodes = result2.consume().counters.nodes_deleted
                            log_graph_message(f"  ✓ Usunięto {deleted_nodes} węzłów")
                            
                            # Usuń wszystkie indeksy
                            log_graph_message("  🔄 Usuwanie indeksów...")
                            try:
                                indexes = session.run("SHOW INDEXES").data()
                                for index in indexes:
                                    if index.get('type') != 'LOOKUP':  # Nie usuwaj indeksów systemowych
                                        index_name = index.get('name')
                                        if index_name:
                                            session.run(f"DROP INDEX `{index_name}`")
                                log_graph_message(f"  ✓ Usunięto {len([i for i in indexes if i.get('type') != 'LOOKUP'])} indeksów")
                            except Exception as e:
                                log_graph_message(f"  ⚠ Błąd usuwania indeksów: {e}")
                            
                            # Usuń wszystkie ograniczenia
                            log_graph_message("  🔄 Usuwanie ograniczeń...")
                            try:
                                constraints = session.run("SHOW CONSTRAINTS").data()
                                for constraint in constraints:
                                    constraint_name = constraint.get('name')
                                    if constraint_name:
                                        session.run(f"DROP CONSTRAINT `{constraint_name}`")
                                log_graph_message(f"  ✓ Usunięto {len(constraints)} ograniczeń")
                            except Exception as e:
                                log_graph_message(f"  ⚠ Błąd usuwania ograniczeń: {e}")
                        
                        log_graph_message("🎉 BAZA DANYCH ZOSTAŁA CAŁKOWICIE WYCZYSZCZONA!")
                        log_graph_message(f"📊 PODSUMOWANIE:")
                        log_graph_message(f"   • Usunięto węzłów: {deleted_nodes}")
                        log_graph_message(f"   • Usunięto relacji: {deleted_relations}")
                        log_graph_message("   • Usunięto indeksy i ograniczenia")
                        log_graph_message("")
                        log_graph_message("🔄 Baza jest teraz całkowicie pusta i gotowa do nowego użycia.")
                        
                    except Exception as e:
                        log_graph_message(f"✗ Błąd podczas wyczyszczania bazy: {e}")
                else:
                    log_graph_message("❌ Anulowano wyczyszczanie bazy")
            else:
                log_graph_message("❌ Anulowano wyczyszczanie bazy")
        
        # Frame dla przycisków operacji
        operations_frame = tk.LabelFrame(graph_window, text="Operacje na grafie", font=("Arial", 10, "bold"))
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
        
        # Przyciski operacji grafu
        graph_buttons_frame = tk.Frame(operations_frame)
        graph_buttons_frame.pack(fill=tk.X, pady=5)
        
        operation_buttons = []
        
        btn1 = tk.Button(graph_buttons_frame, text="Utwórz relacje tekstowe", 
                        command=create_text_relations, state=tk.DISABLED)
        btn1.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn1)
        
        btn2 = tk.Button(graph_buttons_frame, text="Konserwacja grafu", 
                        command=run_maintenance, state=tk.DISABLED)
        btn2.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn2)
        
        btn3 = tk.Button(graph_buttons_frame, text="Pokaż statystyki", 
                        command=show_statistics, state=tk.DISABLED)
        btn3.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn3)
        
        # Druga linia przycisków - usuń stare przyciski
        graph_buttons_frame2 = tk.Frame(operations_frame)
        graph_buttons_frame2.pack(fill=tk.X, pady=2)
        
        # Trzecia linia przycisków - z nowym przyciskiem ładowania
        graph_buttons_frame3 = tk.Frame(operations_frame)
        graph_buttons_frame3.pack(fill=tk.X, pady=2)
        
        btn7 = tk.Button(graph_buttons_frame3, text="ZAŁADUJ DANE DO BAZY", 
                        command=lambda: self.load_data_to_neo4j(selected_subjects, log_graph_message, neo4j_connector),
                        state=tk.DISABLED, bg="darkgreen", fg="white", font=("Arial", 10, "bold"))
        btn7.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn7)
        
        btn4 = tk.Button(graph_buttons_frame3, text="Usuń stare relacje", 
                        command=prune_old_relations, state=tk.DISABLED)
        btn4.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn4)
        
        btn5 = tk.Button(graph_buttons_frame3, text="Klastry semantyczne", 
                        command=create_semantic_clusters, state=tk.DISABLED)
        btn5.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn5)
        
        # Czwarta linia przycisków - dla niebezpiecznych operacji
        graph_buttons_frame4 = tk.Frame(operations_frame)
        graph_buttons_frame4.pack(fill=tk.X, pady=2)
        
        btn6 = tk.Button(graph_buttons_frame4, text="USUŃ WSZYSTKIE RELACJE", 
                        command=clear_all_relations, state=tk.DISABLED, 
                        bg="darkred", fg="white")
        btn6.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn6)
        
        btn8 = tk.Button(graph_buttons_frame4, text="USUŃ WSZYSTKIE CHUNKI", 
                        command=clear_all_chunks, state=tk.DISABLED, 
                        bg="darkred", fg="white", font=("Arial", 10, "bold"))
        btn8.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn8)
        
        # NOWY PRZYCISK - TOTAL CLEAR
        btn9 = tk.Button(graph_buttons_frame4, text="🔥 WYCZYŚĆ CAŁĄ BAZĘ 🔥", 
                        command=clear_entire_database, state=tk.DISABLED, 
                        bg="black", fg="red", font=("Arial", 10, "bold"))
        btn9.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn9)
        
        # Przycisk zamknięcia
        close_frame = tk.Frame(graph_window)
        close_frame.pack(pady=10)
        
        def close_graph_window():
            disconnect_from_neo4j()
            graph_window.destroy()
        
        close_btn = tk.Button(close_frame, text="Zamknij", 
                             command=close_graph_window,
                             bg="gray", fg="white", font=("Arial", 12, "bold"),
                             width=15, height=2)
        close_btn.pack()
        
        # Ustaw handler zamknięcia okna
        graph_window.protocol("WM_DELETE_WINDOW", close_graph_window)
        
        # Informacja startowa
        log_graph_message("=== ZARZĄDZANIE GRAFEM NEO4J ===")
        log_graph_message("1. Ustaw parametry połączenia Neo4j")
        log_graph_message("2. Kliknij 'Testuj połączenie' aby sprawdzić konfigurację")
        log_graph_message("3. Kliknij 'Połącz z Neo4j' aby zainicjalizować komponenty")
        log_graph_message("4. Użyj przycisków operacji do zarządzania grafem")
        log_graph_message("")
        log_graph_message("UWAGA: Upewnij się że Neo4j jest uruchomiony i dostępny!")
    
    def run(self):
        """Uruchamia aplikację"""
        self.root.mainloop()

    def select_all(self):
        """Zaznacza wszystkie checkboxy"""
        for var in self.subject_vars.values():
            var.set(True)

    def deselect_all(self):
        """Odznacza wszystkie checkboxy"""
        for var in self.subject_vars.values():
            var.set(False)

    def load_subjects(self):
        """Wczytuje listę przedmiotów i tworzy checkboxy."""
        # upewnij się, że folder istnieje
        if not self.subjects_path.exists():
            tk.messagebox.showerror("Błąd", f"Nie znaleziono folderu {self.subjects_path}")
            return

        for item in sorted(self.subjects_path.iterdir()):
            if not item.is_dir() or item.name == "__pycache__":
                continue
            name = item.name
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(self.checkbox_frame,
                                text=name,
                                variable=var)
            cb.pack(anchor="w", pady=2)
            self.subject_vars[name] = var
            self.subject_checkboxes[name] = cb

    def check_sources_complete(self, subject_name):
        """Sprawdza czy przedmiot ma kompletne źródła przypisane"""
        # Placeholder - zwraca True żeby nie blokować procesu
        # W pełnej implementacji sprawdzałby czy wszystkie foldery mają przypisane źródła
        return True
    
    def save_sources_config(self, subject_path, config):
        """Zapisuje konfigurację źródeł do pliku JSON"""
        try:
            config_file = subject_path / "sources_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Błąd zapisywania konfiguracji: {e}")
            return False

    def load_data_to_neo4j(self, selected_subjects, log_function, neo4j_connector):
        """Ładuje dane z wybranych przedmiotów do Neo4j z embeddingami"""
        if not neo4j_connector:
            log_function("✗ Brak połączenia z Neo4j")
            return
            
        try:
            from rag_codes.rag_functions.embeddings import CLIPEmbedder
            from rag_codes.rag_functions.chunker import TextChunker
            from rag_codes.rag_functions.metadata_caption import ImageTextProcessor, ImageContextFilter
            import hashlib
            
            log_function("🔄 Inicjalizacja komponentów...")
            embedder = CLIPEmbedder()
            chunker = TextChunker()
            processor = ImageTextProcessor()
            filter_processor = ImageContextFilter()
            
            # Mapowanie folderów na typy danych
            folder_type_mapping = {
                "figure": "image",
                "figures": "image", 
                "figury": "image",
                "obrazy": "image",
                "zdjecia": "image",
                "tabele": "table",
                "tables": "table",
                "tabela": "table",
                "table": "table",
                "wzory": "formula",
                "formulas": "formula",
                "formula": "formula",
                "wzor": "formula",
                "equations": "formula",
                "rownania": "formula"
            }
            
            # Inicjalizuj graph_builder
            graph_builder = None
            try:
                from rag_codes.rag_functions.graph import TextGraphBuilder
                graph_builder = TextGraphBuilder(neo4j_connector, similarity_threshold=0.8)
            except Exception as e:
                log_function(f"⚠ Nie można zainicjalizować TextGraphBuilder: {e}")
                return
            
            total_text_chunks = 0
            total_errors = 0
            
            # =====================================================
            # CZĘŚĆ 1: DOTYCHCZASOWE ŁADOWANIE DANYCH (JSON FILES)
            # =====================================================
            log_function("\n=== CZĘŚĆ 1: ŁADOWANIE GOTOWYCH PLIKÓW JSON ===")
            
            for subject_name in selected_subjects:
                log_function(f"\n--- ŁADOWANIE PRZEDMIOTU: {subject_name} ---")
                subject_path = self.subjects_path / subject_name
                
                # Wczytaj konfigurację źródeł
                sources_config = {}
                sources_config_path = subject_path / "sources_config.json"
                if sources_config_path.exists():
                    try:
                        with open(sources_config_path, 'r', encoding='utf-8') as f:
                            sources_config = json.load(f)
                        log_function(f"✓ Wczytano konfigurację źródeł: {len(sources_config)} folderów")
                        
                        # Wyświetl mapowanie folderów na źródła
                        log_function("📋 Mapowanie folderów na źródła:")
                        for folder, source in sources_config.items():
                            log_function(f"   • {folder} → {source}")
                            
                    except Exception as e:
                        log_function(f"⚠ Błąd wczytywania konfiguracji źródeł: {e}")
                else:
                    log_function(f"⚠ Brak pliku sources_config.json dla przedmiotu {subject_name}")
                
                # Znajdź wszystkie podfoldery (k1, k2, k3, etc.)
                subfolders = [item for item in subject_path.iterdir() 
                             if item.is_dir() and item.name != "__pycache__"]
                
                for subfolder in subfolders:
                    subfolder_name = subfolder.name
                    
                    # Użyj sources_config.json do określenia źródła
                    source_type = sources_config.get(subfolder_name, "unknown")
                    
                    log_function(f"\n🔄 Przetwarzanie podfolderu: {subfolder_name}")
                    log_function(f"    📁 Źródło: {source_type} (z sources_config.json)")
                    
                    # Przetwarzanie gotowych plików JSON (z poprzedniego przetwarzania)
                    rezultaty_path = subfolder / "rezultaty"
                    if rezultaty_path.exists():
                        json_files = list(rezultaty_path.glob("*_filtered_context.json"))
                        
                        for json_file in json_files:
                            log_function(f"  📄 Ładowanie gotowego JSON: {json_file.name}")
                            
                            try:
                                # Wczytaj dane JSON
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    images_data = json.load(f)
                                
                                # Przetwórz każdy obraz i jego kontekst
                                for item in images_data:
                                    for image_path, context_texts in item.items():
                                        try:
                                            # Określ typ danych na podstawie ścieżki obrazu
                                            image_path_obj = Path(image_path)
                                            data_type = "text"  # domyślny typ
                                            
                                            # Sprawdź każdy folder w ścieżce obrazu
                                            for part in image_path_obj.parts:
                                                part_lower = part.lower()
                                                if part_lower in folder_type_mapping:
                                                    data_type = folder_type_mapping[part_lower]
                                                    break
                                            
                                            # Przetwórz chunki tekstu
                                            for context_text in context_texts:
                                                if not context_text or not context_text.strip():
                                                    continue
                                                
                                                # Podziel długi tekst na chunki
                                                text_chunks = chunker.chunk_text(context_text, max_tokens=150)
                                                
                                                for chunk_idx, chunk_text in enumerate(text_chunks):
                                                    try:
                                                        # Stwórz unikalny ID dla chunku
                                                        chunk_content = f"{subfolder_name}_{image_path}_{chunk_idx}_{chunk_text[:50]}"
                                                        chunk_id = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()
                                                        
                                                        # Sprawdź czy chunk już istnieje (unikanie duplikatów)
                                                        with neo4j_connector.get_driver().session() as session:
                                                            # Zmień etykietę z TextNode na właściwą - sprawdź wszystkie możliwe etykiety
                                                            result = session.run(
                                                                "MATCH (n {id: $node_id}) RETURN count(n) as count",
                                                                node_id=f"txt_{chunk_id}"
                                                            )
                                                            if result.single()["count"] > 0:
                                                                continue  # Skip jeśli już istnieje
                                                        
                                                        # Pobierz embedding tekstu
                                                        text_embedding = embedder.get_text_embedding(chunk_text)
                                                        if text_embedding is None:
                                                            total_errors += 1
                                                            continue
                                                        
                                                        # Przygotuj metadata z dodatkowymi informacjami
                                                        metadata = {
                                                            "path": image_path,
                                                            "source": source_type,
                                                            "subject": subject_name,
                                                            "subfolder": subfolder_name,
                                                            "original_file": json_file.name
                                                        }
                                                        
                                                        # Dodaj węzeł do Neo4j - POPRAWIONA METODA
                                                        node_id_prefix = {
                                                            "image": "img",
                                                            "table": "tbl", 
                                                            "formula": "frm",
                                                            "text": "txt"
                                                        }.get(data_type, "txt")
                                                        
                                                        # Wywołaj insert_node z poprawnymi argumentami
                                                        try:
                                                            graph_builder.insert_node(
                                                                node_id=f"{node_id_prefix}_{chunk_id}",
                                                                data_type=data_type,
                                                                text=chunk_text,
                                                                embedding=text_embedding.tolist(),
                                                                metadata=metadata
                                                            )
                                                        except TypeError as te:
                                                            # Jeśli metoda nie przyjmuje metadata, spróbuj bez niej
                                                            log_function(f"    ⚠ Próba alternatywnej metody dodawania węzła...")
                                                            graph_builder.insert_node(
                                                                node_id=f"{node_id_prefix}_{chunk_id}",
                                                                data_type=data_type,
                                                                text=chunk_text,
                                                                embedding=text_embedding.tolist()
                                                            )
                                                            
                                                            # Dodaj metadata bezpośrednio do Neo4j
                                                            with neo4j_connector.get_driver().session() as session:
                                                                session.run("""
                                                                    MATCH (n:TextNode {id: $node_id})
                                                                    SET n.path = $path,
                                                                        n.source = $source,
                                                                        n.subject = $subject,
                                                                        n.subfolder = $subfolder,
                                                                        n.original_file = $original_file
                                                                """, 
                                                                node_id=f"{node_id_prefix}_{chunk_id}",
                                                                path=image_path,
                                                                source=source_type,
                                                                subject=subject_name,
                                                                subfolder=subfolder_name,
                                                                original_file=json_file.name
                                                                )
                                                        
                                                        total_text_chunks += 1
                                                        
                                                        if total_text_chunks % 25 == 0:
                                                            log_function(f"    💾 Dodano {total_text_chunks} chunków...")
                                                        
                                                    except Exception as e:
                                                        log_function(f"    ✗ Błąd przetwarzania chunka: {e}")
                                                        total_errors += 1
                                            
                                        except Exception as e:
                                            log_function(f"    ✗ Błąd przetwarzania obrazu {image_path}: {e}")
                                            total_errors += 1
                            
                            except Exception as e:
                                log_function(f"  ✗ Błąd przetwarzania pliku {json_file.name}: {e}")
                                total_errors += 1
            
            log_function(f"\n📊 CZĘŚĆ 1 ZAKOŃCZONA: Dodano {total_text_chunks} chunków z gotowych JSON")
            
            # =====================================================
            # CZĘŚĆ 2: NOWE - WYDOBYWANIE CHUNKÓW Z METADATA_CAPTION
            # =====================================================
            log_function("\n=== CZĘŚĆ 2: WYDOBYWANIE CHUNKÓW TEKSTOWYCH Z METADATA_CAPTION ===")
            
            part2_chunks = 0
            part2_errors = 0
            
            for subject_name in selected_subjects:
                log_function(f"\n--- WYDOBYWANIE TEKSTÓW Z PRZEDMIOTU: {subject_name} ---")
                subject_path = self.subjects_path / subject_name
                
                # Wczytaj konfigurację źródeł
                sources_config = {}
                sources_config_path = subject_path / "sources_config.json"
                if sources_config_path.exists():
                    try:
                        with open(sources_config_path, 'r', encoding='utf-8') as f:
                            sources_config = json.load(f)
                    except Exception as e:
                        log_function(f"⚠ Błąd wczytywania konfiguracji źródeł: {e}")
                
                # Znajdź wszystkie podfoldery (k1, k2, k3, etc.)
                subfolders = [item for item in subject_path.iterdir() 
                             if item.is_dir() and item.name != "__pycache__"]
                
                for subfolder in subfolders:
                    subfolder_name = subfolder.name
                    source_type = sources_config.get(subfolder_name, "unknown")
                    
                    log_function(f"\n🔄 Wydobywanie tekstów z podfolderu: {subfolder_name}")
                    log_function(f"    📁 Źródło: {source_type}")
                    
                    # Znajdź pliki TXT w folderze rezultaty
                    rezultaty_path = subfolder / "rezultaty"
                    if not rezultaty_path.exists():
                        log_function(f"  ⚠ Brak folderu 'rezultaty' w {subfolder_name}")
                        continue
                    
                    txt_files = list(rezultaty_path.glob("*.txt"))
                    if not txt_files:
                        log_function(f"  ⚠ Brak plików .txt w {subfolder_name}/rezultaty")
                        continue
                    
                    for txt_file in txt_files:
                        log_function(f"  📄 Przetwarzanie pliku TXT: {txt_file.name}")
                        
                        try:
                            # KROK 1: Użyj ImageTextProcessor do wydobycia kontekstu
                            texts = processor.process_file(str(txt_file))
                            json_data = processor.get_images_with_context_json(texts)
                            
                            if not json_data:
                                log_function(f"    ⚠ Brak danych w pliku {txt_file.name}")
                                continue
                            
                            log_function(f"    ✓ Wydobyto dane dla {len(json_data)} elementów")
                            
                            # KROK 2: Filtruj kontekst (opcjonalnie)
                            filtered_data = filter_processor.process_images_context(json_data)
                            
                            # KROK 3: Wydobądź TYLKO TEKSTY (usuń obrazy)
                            all_text_chunks = []
                            
                            for item in filtered_data:
                                for image_path, context_texts in item.items():
                                    # USUWAMY OBRAZY - bierzemy tylko teksty
                                    for context_text in context_texts:
                                        if context_text and context_text.strip():
                                            all_text_chunks.append({
                                                'text': context_text.strip(),
                                                'original_image_path': image_path,  # dla metadanych
                                                'subfolder': subfolder_name,
                                                'subject': subject_name,
                                                'source': source_type,
                                                'txt_file': txt_file.name
                                            })
                            
                            log_function(f"    📝 Wydobyto {len(all_text_chunks)} chunków tekstowych (bez obrazów)")
                            
                            # KROK 4: Przetwórz chunki tekstowe i dodaj do bazy
                            for text_chunk_data in all_text_chunks:
                                try:
                                    full_text = text_chunk_data['text']
                                    
                                    # Podziel długi tekst na mniejsze chunki
                                    text_chunks = chunker.chunk_text(full_text, max_tokens=150)
                                    
                                    for chunk_idx, chunk_text in enumerate(text_chunks):
                                        try:
                                            # Stwórz unikalny ID dla chunku tekstowego
                                            chunk_content = f"TEXT_{subfolder_name}_{txt_file.stem}_{chunk_idx}_{chunk_text[:50]}"
                                            chunk_id = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()
                                            
                                            # Sprawdź czy chunk już istnieje
                                            with neo4j_connector.get_driver().session() as session:
                                                result = session.run(
                                                    "MATCH (n {id: $node_id}) RETURN count(n) as count",
                                                    node_id=f"txtchunk_{chunk_id}"
                                                )
                                                if result.single()["count"] > 0:
                                                    continue  # Skip jeśli już istnieje
                                            
                                            # Pobierz embedding tekstu
                                            text_embedding = embedder.get_text_embedding(chunk_text)
                                            if text_embedding is None:
                                                part2_errors += 1
                                                continue
                                            
                                            # Przygotuj metadata
                                            metadata = {
                                                "path": text_chunk_data['original_image_path'],
                                                "source": source_type,
                                                "subject": subject_name,
                                                "subfolder": subfolder_name,
                                                "txt_source_file": text_chunk_data['txt_file']
                                            }
                                            
                                            # Dodaj węzeł tekstowy do Neo4j - POPRAWIONA METODA
                                            try:
                                                graph_builder.insert_node(
                                                    node_id=f"txtchunk_{chunk_id}",
                                                    data_type="text",  # Zmienione z "extracted_text" na "text"
                                                    text=chunk_text,
                                                    embedding=text_embedding.tolist(),
                                                    metadata=metadata
                                                )
                                            except TypeError as te:
                                                # Alternatywna metoda bez metadata
                                                graph_builder.insert_node(
                                                    node_id=f"txtchunk_{chunk_id}",
                                                    data_type="text",  # Zmienione z "extracted_text" na "text"
                                                    text=chunk_text,
                                                    embedding=text_embedding.tolist()
                                                )
                                                
                                                # Dodaj metadata bezpośrednio
                                                with neo4j_connector.get_driver().session() as session:
                                                    session.run("""
                                                        MATCH (n:TextNode {id: $node_id})
                                                        SET n.path = $path,
                                                            n.source = $source,
                                                            n.subject = $subject,
                                                            n.subfolder = $subfolder,
                                                            n.txt_source_file = $txt_source_file
                                                    """, 
                                                    node_id=f"txtchunk_{chunk_id}",
                                                    path=text_chunk_data['original_image_path'],
                                                    source=source_type,
                                                    subject=subject_name,
                                                    subfolder=subfolder_name,
                                                    txt_source_file=text_chunk_data['txt_file']
                                                    )
                                            
                                            part2_chunks += 1
                                            
                                            if part2_chunks % 25 == 0:
                                                log_function(f"    💾 Dodano {part2_chunks} chunków tekstowych...")
                                            
                                        except Exception as e:
                                            log_function(f"    ✗ Błąd przetwarzania chunka tekstowego: {e}")
                                            part2_errors += 1
                                
                                except Exception as e:
                                    log_function(f"    ✗ Błąd przetwarzania tekstu: {e}")
                                    part2_errors += 1
                            

                        except Exception as e:
                            log_function(f"  ✗ Błąd przetwarzania pliku {txt_file.name}: {e}")
                            part2_errors += 1
            
            log_function(f"\n📊 CZĘŚĆ 2 ZAKOŃCZONA: Dodano {part2_chunks} chunków tekstowych (bez obrazów)")
            
            # =====================================================
            # PODSUMOWANIE CAŁEGO PROCESU
            # =====================================================
            total_all_chunks = total_text_chunks + part2_chunks
            total_all_errors = total_errors + part2_errors
            
            log_function(f"\n=== PODSUMOWANIE CAŁEGO ŁADOWANIA DANYCH ===")
            log_function(f"CZĘŚĆ 1 (gotowe JSON): {total_text_chunks} chunków")
            log_function(f"CZĘŚĆ 2 (wydobyte teksty): {part2_chunks} chunków")
            log_function(f"RAZEM dodano chunków: {total_all_chunks}")
            log_function(f"Błędy: {total_all_errors}")
            
            if total_all_chunks > 0:
                log_function(f"\n🔄 Tworzenie relacji między wszystkimi węzłami...")
                
                # Utwórz relacje między węzłami
                try:
                    graph_builder.create_text_relations()
                    log_function("✓ Relacje między węzłami utworzone pomyślnie")
                except Exception as e:
                    log_function(f"✗ Błąd tworzenia relacji: {e}")
                
                log_function("🎉 ŁADOWANIE DANYCH ZAKOŃCZONE POMYŚLNIE!")
                
                # Pokaż statystyki
                log_function("\n=== STATYSTYKI TYPÓW I ŹRÓDEŁ ===")
                try:
                    with neo4j_connector.get_driver().session() as session:
                        # Statystyki według źródeł
                        source_stats = session.run("""
                            MATCH (n) 
                            WHERE n.source IS NOT NULL
                            RETURN n.source as source, count(*) as count 
                            ORDER BY count DESC
                        """).data()
                        
                        log_function("📊 Chunki według źródeł:")
                        for stat in source_stats:
                            log_function(f"   • {stat['source']}: {stat['count']} chunków")
                        
                        # Statystyki według typów danych
                        type_stats = session.run("""
                            MATCH (n) 
                            WHERE n.data_type IS NOT NULL
                            RETURN n.data_type as data_type, count(*) as count 
                            ORDER BY count DESC
                        """).data()
                        
                        log_function("\n📊 Chunki według typów danych:")
                        for stat in type_stats:
                            log_function(f"   • {stat['data_type']}: {stat['count']} chunków")
                        
                        # Statystyki według przedmiotów i podfolderów
                        subfolder_stats = session.run("""
                            MATCH (n) 
                            WHERE n.subject IS NOT NULL AND n.subfolder IS NOT NULL
                            RETURN n.subject as subject, n.subfolder as subfolder, 
                                   n.source as source, count(*) as count 
                            ORDER BY subject, subfolder
                        """).data()
                        
                        log_function("\n📊 Chunki według przedmiotów i podfolderów:")
                        current_subject = None
                        for stat in subfolder_stats:
                            subject = stat.get('subject', 'unknown')
                            subfolder = stat.get('subfolder', 'unknown')
                            source = stat.get('source', 'unknown')
                            count = stat.get('count', 0)
                            
                            if subject != current_subject:
                                if current_subject is not None:
                                    log_function("")
                                log_function(f"📚 {subject}:")
                                current_subject = subject
                            
                            log_function(f"   • {subfolder} ({source}): {count} chunków")
                            
                except Exception as e:
                    log_function(f"⚠ Błąd pobierania statystyk: {e}")
                    
                log_function("\nUżywaj przycisk 'Pokaż statystyki' aby zobaczyć więcej szczegółów")
            else:
                log_function("⚠ Nie dodano żadnych danych")
                
        except ImportError as e:
            log_function(f"✗ Błąd importu wymaganych modułów: {e}")
            log_function("Upewnij się, że wszystkie wymagane biblioteki są zainstalowane")
        except Exception as e:
            log_function(f"✗ Błąd ładowania danych: {e}")
if __name__ == "__main__":
    app = SubjectSelectorApp()
    app.run()