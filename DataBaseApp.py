import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import json
import threading
import hashlib
import torch
import time

RELATION_SIMILARITY_THRESHOLD = 0.985 
MAX_TOKENS = 150

from rag_codes.rag_functions.metadata_context import ImageTextProcessor
from rag_codes.rag_functions.embeddings import CLIPEmbedder, CohereEmbedder
from rag_codes.rag_functions.graph import (
    GraphBuilder, Neo4jConnector
)

class SubjectSelectorApp:
    def __init__(self):
        """
        Inicjalizuje aplikację, ustawia główne okno Tkinter, ścieżki,
        zmienne kontrolne oraz tworzy widżety i wczytuje przedmioty.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
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
        
        # NOWE: Zmienna kontrolująca use_vision
        self.use_vision = tk.BooleanVar(value=False)  # Domyślnie False
        
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
        """
        Tworzy wszystkie elementy interfejsu użytkownika (UI) w głównym oknie,
        w tym sekcję Vision, listę przedmiotów, przyciski kontrolne oraz akcje
        do przejścia dalej lub zamknięcia aplikacji.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
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
        
        # NOWE: Frame dla kontroli Vision
        vision_control_frame = tk.Frame(self.root, bg="lightyellow", relief=tk.RAISED, bd=2)
        vision_control_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Tytuł sekcji Vision
        vision_title = tk.Label(vision_control_frame, 
                               text="🔍 Kontrola Vision (przetwarzanie obrazów)", 
                               font=("Arial", 11, "bold"), bg="lightyellow")
        vision_title.pack(pady=5)
        
        # Checkbox dla use_vision
        vision_checkbox_frame = tk.Frame(vision_control_frame, bg="lightyellow")
        vision_checkbox_frame.pack(pady=5)
        
        self.vision_checkbox = tk.Checkbutton(vision_checkbox_frame, 
                                             text="Użyj Vision API do analizy obrazów", 
                                             variable=self.use_vision,
                                             command=self.on_vision_toggle,
                                             font=("Arial", 10),
                                             bg="lightyellow")
        self.vision_checkbox.pack(side=tk.LEFT)
        
        # Status Vision
        self.vision_status_label = tk.Label(vision_checkbox_frame, 
                                           text="❌ Vision wyłączone - tylko metadane obrazów", 
                                           font=("Arial", 9), fg="red", bg="lightyellow")
        self.vision_status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Informacja o Vision
        vision_info = tk.Label(vision_control_frame, 
                              text="Włącz dla pełnej analizy obrazów z AI (wolniej, ale dokładniej)", 
                              font=("Arial", 8), fg="darkblue", bg="lightyellow")
        vision_info.pack(pady=(0, 5))
        
        # Frame z scrollbarem dla listy przedmiotów
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 5))
        
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
            """
            Aktualizuje obszar przewijania (scrollregion) canvasu na podstawie
            rozmiaru wewnętrznego frame'a (używane do dynamicznego dostosowania
            scrollbar'a po zmianie zawartości).

            Args:
                event (tk.Event): Obiekt zdarzenia wywołujący aktualizację (np. <Configure>).

            Returns:
                None
            """
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except tk.TclError:
                # Widget został zniszczony
                pass
        
        self.checkbox_frame.bind("<Configure>", configure_scroll_region)
        
        # Przyciski kontrolne (Zaznacz/Odznacz wszystko)
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Zaznacz wszystko", 
                 command=self.select_all).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Odznacz wszystko", 
                 command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        
        # Frame dla przycisków głównych - ZAMKNIJ i DALEJ
        main_buttons_frame = tk.Frame(self.root)
        main_buttons_frame.pack(pady=10, side=tk.BOTTOM)
        
        # Przycisk ZAMKNIJ - zamyka program
        tk.Button(main_buttons_frame, 
                 text="ZAMKNIJ", 
                 command=self.close_application,
                 bg="red", 
                 fg="white",
                 font=("Arial", 12, "bold"),
                 width=12,
                 height=2).pack(side=tk.LEFT, padx=10)
        
        # Przycisk DALEJ - zapisuje JSONy i kontynuuje
        tk.Button(main_buttons_frame, 
                 text="DALEJ", 
                 command=self.save_and_proceed,
                 bg="green", 
                 fg="white",
                 font=("Arial", 12, "bold"),
                 width=12,
                 height=2).pack(side=tk.LEFT, padx=10)
        
        # Przycisk tworzenia plików TXT (chunks + base64)
        tk.Button(main_buttons_frame,
                 text="UTWÓRZ PLIKI TXT",
                 command=self.create_txt_files_only,
                 bg="orange",
                 fg="white",
                 font=("Arial", 12, "bold"),
                 width=15,
                 height=2).pack(side=tk.LEFT, padx=10)

        # POPRAWKA: Tylko jeden przycisk zarządzania bazą danych
        tk.Button(main_buttons_frame,
                 text="Zarządzanie bazą danych",
                 command=lambda: self.open_graph_management_window(list(self.subject_vars.keys())),
                 bg="blue",
                 fg="white",
                 font=("Arial", 12, "bold"),
                 width=20,
                 height=2).pack(side=tk.LEFT, padx=10)
        

    def on_vision_toggle(self):
        """
        Obsługuje zmianę stanu opcji Vision (checkbox).
        Aktualizuje etykietę statusu Vision w UI.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
        if self.use_vision.get():
            self.vision_status_label.config(text="✅ Vision włączone - pełna analiza obrazów", fg="green")
        else:
            self.vision_status_label.config(text="❌ Vision wyłączone - tylko metadane obrazów", fg="red")
    
    def load_subjects(self):
        """
        Wczytuje listę przedmiotów z folderu `subjects` i tworzy dla nich checkboxy
        oraz przyciski do konfiguracji źródeł.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
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
        """
        Zaznacza wszystkie checkboxy w liście przedmiotów.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
        for var in self.subject_vars.values():
            var.set(True)

    def deselect_all(self):
        """
        Odznacza wszystkie checkboxy w liście przedmiotów.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
        for var in self.subject_vars.values():
            var.set(False)

    def close_application(self):
        """
        Zamyka aplikację po potwierdzeniu użytkownika.
        Ostrzega o niezapisanych zmianach w konfiguracji źródeł.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
        message = "Czy na pewno chcesz zamknąć aplikację?"
        if self.temp_sources_configs:
            message += "\nMasz niezapisane zmiany w źródłach, które zostaną utracone!"
        
        if messagebox.askyesno("Potwierdzenie zamknięcia", message):
            self.root.destroy()

    def save_and_proceed(self):
        """
        Zapisuje konfiguracje źródeł dla zaznaczonych przedmiotów,
        a następnie przechodzi do okna przetwarzania danych.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
        selected_subjects = [subject for subject, var in self.subject_vars.items() if var.get()]
        
        if not selected_subjects:
            messagebox.showwarning("Brak wyboru", 
                                  "Nie wybrano żadnego przedmiotu!\n"
                                  "Zaznacz przynajmniej jeden przedmiot przed kontynuowaniem.")
            return

        # Sprawdzenie źródeł (placeholder - zawsze True)

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
    
    def save_sources_config(self, subject_path, config):
        """
        Zapisuje konfigurację źródeł dla wybranego przedmiotu do pliku JSON.

        Args:
            subject_path (Path): Ścieżka do folderu przedmiotu.
            config (dict): Mapowanie folderów na typy źródeł.

        Returns:
            bool: True, jeśli zapis zakończył się sukcesem, False w przypadku błędu.
        """
        try:
            config_file = subject_path / "sources_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Zapisano konfigurację źródeł do {config_file}")
            return True
        except Exception as e:
            print(f"❌ Błąd zapisywania konfiguracji źródeł: {e}")
            return False

    def open_source_config(self, subject_name):
        """
        Otwiera okno konfiguracji źródeł dla podfolderów danego przedmiotu.
        Pozwala ustawić typ źródła dla każdego podfolderu i zapisać zmiany.

        Args:
            subject_name (str): Nazwa przedmiotu.

        Returns:
            None
        """
        subject_path = self.subjects_path / subject_name
        
        # Wczytaj istniejącą konfigurację
        existing_config = {}
        config_file = subject_path / "sources_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                print(f"📄 Wczytano istniejącą konfigurację z {config_file}")
            except Exception as e:
                print(f"⚠️ Błąd wczytywania istniejącej konfiguracji: {e}")
        
        # Sprawdź czy są tymczasowe zmiany
        temp_config = self.temp_sources_configs.get(subject_name, {})
        # Połącz istniejącą konfigurację z tymczasową (tymczasowa ma priorytet)
        merged_config = {**existing_config, **temp_config}

        win = tk.Toplevel(self.root)
        win.title(f"Źródła: {subject_name}")
        win.geometry("500x400")
        win.grab_set()

        vars_map = {}
        tk.Label(win, text=f"Typ źródła dla folderów {subject_name}",
                font=("Arial", 12, "bold")).pack(pady=10)

        # Info o zapisie
        info_label = tk.Label(win, 
                            text="💾 Konfiguracja zostanie zapisana do pliku sources_config.json", 
                            font=("Arial", 9), fg="blue")
        info_label.pack(pady=5)

        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Sprawdź podfoldery
        subfolders = []
        if subject_path.exists():
            for sub in sorted(subject_path.iterdir()):
                if sub.is_dir() and sub.name != "__pycache__":
                    subfolders.append(sub)

        if not subfolders:
            tk.Label(frame, text="Brak podfolderów do konfiguracji", 
                    font=("Arial", 10), fg="red").pack(pady=20)
        else:
            for sub in subfolders:
                row = tk.Frame(frame)
                row.pack(fill=tk.X, pady=2)
                
                tk.Label(row, text=sub.name, width=20, anchor="w").pack(side=tk.LEFT)

                # Użyj wartości z merged_config lub domyślną "unknown"
                current_value = merged_config.get(sub.name, "unknown")
                var = tk.StringVar(value=current_value)
                menu = ttk.OptionMenu(row, var, current_value, *self.source_types)
                menu.pack(side=tk.LEFT, padx=5)
                vars_map[sub.name] = var
        # DODANA FUNKCJA ZAPISYWANIA I ZAMYKANIA
        def save_and_close():
            """
            Zapisuje konfigurację źródeł dla podfolderów danego przedmiotu
            do pliku `sources_config.json` oraz zamyka okno konfiguracji.

            Args:
                None

            Returns:
                None
            """
            try:
                # Pobierz wartości z wszystkich pól
                new_config = {sub_name: var.get() for sub_name, var in vars_map.items()}
                
                # Zapisz do tymczasowej konfiguracji
                self.temp_sources_configs[subject_name] = new_config
                
                # Sprawdź czy ma być zapisane od razu do pliku
                config_file = subject_path / "sources_config.json"
                try:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(new_config, f, indent=2, ensure_ascii=False)
                    print(f"✅ Zapisano konfigurację źródeł do {config_file}")
                    messagebox.showinfo("Sukces", 
                        f"Konfiguracja źródeł została zapisana do pliku:\n{config_file.name}\n\n"
                        f"Zapisano {len(new_config)} folderów.")
                except Exception as e:
                    print(f"❌ Błąd zapisywania do pliku: {e}")
                    # Mimo błędu zapisu do pliku, zachowaj w temp_sources_configs
                    messagebox.showwarning("Częściowy sukces", 
                        f"Konfiguracja została zachowana tymczasowo, ale wystąpił błąd zapisu do pliku:\n{e}\n\n"
                        f"Konfiguracja zostanie zapisana przy kontynuowaniu.")
                
                win.destroy()
                
            except Exception as e:
                messagebox.showerror("Błąd", f"Błąd zapisywania konfiguracji: {e}")

        def cancel():
            """
            Anuluje wprowadzane zmiany w konfiguracji źródeł
            i zamyka okno konfiguracji bez zapisywania.

            Args:
                None

            Returns:
                None
            """
            win.destroy()

        # DODANE PRZYCISKI
        button_frame = tk.Frame(win)
        button_frame.pack(pady=10, side=tk.BOTTOM)
        
        # Przycisk Zapisz i zamknij
        save_btn = tk.Button(button_frame, 
                            text="Zapisz i zamknij", 
                            command=save_and_close,
                            bg="green", 
                            fg="white",
                            font=("Arial", 10, "bold"),
                            width=15)
        save_btn.pack(side=tk.LEFT, padx=10)
        
        # Przycisk Anuluj
        cancel_btn = tk.Button(button_frame, 
                            text="Anuluj", 
                            command=cancel,
                            bg="red", 
                            fg="white",
                            font=("Arial", 10, "bold"),
                            width=10)
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # DODANE: Info o liczbie folderów
        if subfolders:
            info_folders = tk.Label(win, 
                                text=f"Konfiguracja dla {len(subfolders)} folderów", 
                                font=("Arial", 8), fg="gray")
            info_folders.pack(pady=2, side=tk.BOTTOM)
    def open_processing_window(self, selected_subjects):
        """
        Tworzy i otwiera nowe okno do przetwarzania wybranych przedmiotów,
        z logiem, paskiem postępu i przyciskiem do uruchomienia przetwarzania.

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.

        Returns:
            None
        """
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


    def create_txt_files_only(self):
        """
        Tworzy jedynie pliki TXT (chunks i base64) dla wybranych przedmiotów,
        bez generowania plików JSON.

        Args:
            self: Instancja klasy SubjectSelectorApp.

        Returns:
            None
        """
        selected_subjects = [subject for subject, var in self.subject_vars.items() if var.get()]
        
        if not selected_subjects:
            messagebox.showwarning("Brak wyboru", 
                                  "Nie wybrano żadnego przedmiotu!\n"
                                  "Zaznacz przynajmniej jeden przedmiot przed kontynuowaniem.")
            return


        # Zapisywanie konfiguracji tymczasowych
        saved_count = 0
        failed_subjects = []
        
        for subject_name in selected_subjects:
            subject_path = self.subjects_path / subject_name
            
            if subject_name in self.temp_sources_configs:
                if self.save_sources_config(subject_path, self.temp_sources_configs[subject_name]):
                    saved_count += 1
                else:
                    failed_subjects.append(subject_name)
        
        if failed_subjects:
            messagebox.showerror("Błąd zapisywania", 
                f"Nie udało się zapisać konfiguracji dla przedmiotów:\n\n" + 
                "\n".join(f"• {subject}" for subject in failed_subjects))
            return

        # Otwórz okno przetwarzania TXT
        self.open_txt_processing_window(selected_subjects)

    def open_txt_processing_window(self, selected_subjects):
        """
        Otwiera okno do przetwarzania tylko plików TXT (chunks + base64)
        dla wybranych przedmiotów.

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.

        Returns:
            None
        """
        processing_window = tk.Toplevel(self.root)
        processing_window.title("Tworzenie plików TXT (chunks + base64)")
        processing_window.geometry("900x700")
        processing_window.grab_set()
        
        # Tytuł
        tk.Label(processing_window, text="Tworzenie plików TXT: chunks + base64", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Info o przedmiatch
        tk.Label(processing_window, 
                text=f"Wybrane przedmioty ({len(selected_subjects)}): " + ", ".join(selected_subjects), 
                font=("Arial", 10), wraplength=800).pack(pady=5)
        
        # Info o plikach
        tk.Label(processing_window, 
                text="📄 Tworzę tylko pliki: *_chunks.txt i *_base64.txt (bez JSON)", 
                font=("Arial", 10), fg="blue", wraplength=800).pack(pady=5)
        
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
        
        start_btn = tk.Button(button_frame, text="ROZPOCZNIJ TWORZENIE TXT", 
                             command=lambda: self.start_txt_processing(selected_subjects, log_text, 
                                                                      progress_bar, progress_label, 
                                                                      start_btn, close_btn),
                             bg="orange", fg="white", font=("Arial", 12, "bold"),
                             width=25, height=2)
        start_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = tk.Button(button_frame, text="ZAMKNIJ", 
                             command=processing_window.destroy,
                             bg="red", fg="white", font=("Arial", 12, "bold"),
                             width=15, height=2)
        close_btn.pack(side=tk.LEFT, padx=10)

    def start_txt_processing(self, selected_subjects, log_text, progress_bar, progress_label, start_btn, close_btn):
        """
        Uruchamia proces tworzenia plików TXT w osobnym wątku, aby nie blokować UI.

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.
            log_text (tk.Text): Widget logu do wyświetlania komunikatów.
            progress_bar (ttk.Progressbar): Pasek postępu.
            progress_label (tk.Label): Etykieta statusu postępu.
            start_btn (tk.Button): Przycisk startowy, który zostanie zablokowany.
            close_btn (tk.Button): Przycisk do zamknięcia okna.

        Returns:
            None
        """
        start_btn.config(state=tk.DISABLED)
        
        def process_in_thread():
            """
            Cel wątku pomocniczego uruchamianego przez start_txt_processing.
            Wywołuje metodę przetwarzającą pliki TXT (process_txt_files_only) korzystając
            ze zmiennych zewnętrznych (selected_subjects, log_text, progress_bar, progress_label)
            i po zakończeniu odblokowuje przycisk startu.

            Args:
                None

            Returns:
                None
            """
            self.process_txt_files_only(selected_subjects, log_text, progress_bar, progress_label)
            start_btn.config(state=tk.NORMAL, text="ZAKOŃCZONO", bg="gray")
        
        processing_thread = threading.Thread(target=process_in_thread)
        processing_thread.daemon = True
        processing_thread.start()

    def process_txt_files_only(self, selected_subjects, log_text, progress_bar, progress_label):
        """
        Przetwarza wybrane przedmioty tworząc tylko pliki TXT: 
        *_chunks.txt oraz *_base64.txt (bez JSON).

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.
            log_text (tk.Text): Widget logu.
            progress_bar (ttk.Progressbar): Pasek postępu.
            progress_label (tk.Label): Etykieta statusu.

        Returns:
            None
        """
        self.log_message(log_text, "=== ROZPOCZĘCIE TWORZENIA PLIKÓW TXT ===")
        
        # Inicjalizacja procesorów
        try:
            self.log_message(log_text, "Inicjalizacja procesorów...")
            processor = ImageTextProcessor()
            self.log_message(log_text, "✓ Procesory zainicjalizowane pomyślnie")
        except Exception as e:
            self.log_message(log_text, f"✗ Błąd inicjalizacji procesorów: {e}")
            return
        
        # Liczenie folderów do przetworzenia
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
        chunks_created = 0
        base64_created = 0
        
        # Przetwarzanie
        for subject_name in selected_subjects:
            self.log_message(log_text, f"\n--- PRZETWARZANIE PRZEDMIOTU: {subject_name} ---")
            
            ocr_folders = subject_ocr_folders[subject_name]
            
            for ocr_folder in ocr_folders:
                processed_count += 1
                self.update_progress(progress_bar, progress_label, processed_count, total_ocr_folders, 
                                f"{subject_name}/{ocr_folder.name}")
                
                self.log_message(log_text, f"\nPrzetwarzanie folderu OCR: {subject_name}/{ocr_folder.name}")
                
                # NOWE: Sprawdź i skopiuj pliki z rezultaty do detekcje jeśli potrzeba
                self.log_message(log_text, f"  🔍 Sprawdzanie plików źródłowych...")
                files_ready = self.copy_required_files_to_detekcje(ocr_folder, log_text)
                
                if not files_ready:
                    self.log_message(log_text, f"  ⚠️ Brak wymaganych plików dla {ocr_folder.name} - pomijam")
                    continue
                
                # Definiuj ścieżkę do folderu detekcje
                detekcje_path = ocr_folder / "detekcje"
                
                # Przetwarzanie tylko pliku txt o nazwie podfolderu
                expected_txt_file = detekcje_path / f"{ocr_folder.name}.txt"
                if not expected_txt_file.exists():
                    self.log_message(log_text, f"  ⚠️ Brak pliku {ocr_folder.name}.txt w {ocr_folder.name}/detekcje")
                    continue
                
                txt_files = [expected_txt_file]  # Lista z jednym plikiem
                
                # Przetwarzanie plików txt
                for txt_file in txt_files:
                    try:
                        self.log_message(log_text, f"  📄 Przetwarzanie pliku: {txt_file.name}")
                        
                        # KROK 1: Przetwórz plik i pobierz texts
                        texts = processor.process_file(str(txt_file))
                        self.log_message(log_text, f"    📋 Znaleziono {len(texts)} elementów do przetworzenia")
                        
                        # KROK 2: Utwórz plik TXT z samymi chunkami
                        chunks_output_file = detekcje_path / f"{txt_file.stem}_chunks.txt"
                        chunks_result = processor.create_output_txt_chunks_only(texts, str(chunks_output_file))
                        
                        if chunks_result:
                            self.log_message(log_text, f"    ✓ Zapisano chunks TXT: {chunks_output_file.name}")
                            chunks_created += 1
                        else:
                            self.log_message(log_text, f"    ✗ Błąd zapisywania chunks TXT: {chunks_output_file.name}")
                        
                        # KROK 3: Utwórz plik TXT z base64
                        base64_output_file = detekcje_path / f"{txt_file.stem}_base64.txt"
                        base64_result = processor.create_output_txt_with_base64(texts, str(base64_output_file))
                        
                        if base64_result:
                            self.log_message(log_text, f"    ✓ Zapisano base64 TXT: {base64_output_file.name}")
                            base64_created += 1
                        else:
                            self.log_message(log_text, f"    ✗ Błąd zapisywania base64 TXT: {base64_output_file.name}")
                        
                        # Zlicz sukces jeśli przynajmniej jeden plik został utworzony
                        if chunks_result or base64_result:
                            success_count += 1
                        else:
                            error_count += 1
                        
                    except Exception as e:
                        self.log_message(log_text, f"    ✗ Błąd przetwarzania {txt_file.name}: {e}")
                        error_count += 1
        
        # Podsumowanie
        self.log_message(log_text, f"\n=== PODSUMOWANIE TWORZENIA PLIKÓW TXT ===")
        self.log_message(log_text, f"Przetworzono przedmiotów: {len(selected_subjects)}")
        self.log_message(log_text, f"Przetworzono folderów OCR: {processed_count}")
        self.log_message(log_text, f"Utworzono plików *_chunks.txt: {chunks_created}")
        self.log_message(log_text, f"Utworzono plików *_base64.txt: {base64_created}")
        self.log_message(log_text, f"Pomyślnie przetworzone foldery: {success_count}")
        self.log_message(log_text, f"Błędy: {error_count}")
        
        if error_count == 0:
            self.log_message(log_text, "🎉 WSZYSTKIE PLIKI TXT UTWORZONE POMYŚLNIE!")
            self.log_message(log_text, "📋 Utworzono dla każdego pliku:")
            self.log_message(log_text, f"  • {chunks_created} plików chunks (*_chunks.txt)")
            self.log_message(log_text, f"  • {base64_created} plików base64 (*_base64.txt)")
            self.log_message(log_text, "\n💡 Pliki TXT gotowe do użycia!")
            self.log_message(log_text, "🔗 Możesz teraz użyć przycisku 'DALEJ' aby utworzyć JSON")
            self.log_message(log_text, "   lub przejść bezpośrednio do zarządzania bazą danych")
        else:
            self.log_message(log_text, f"⚠ ZAKOŃCZONO Z {error_count} BŁĘDAMI")
        
        self.update_progress(progress_bar, progress_label, total_ocr_folders, total_ocr_folders, "Zakończono")

    def start_processing(self, selected_subjects, log_text, progress_bar, progress_label, start_btn, close_btn):
        """
        Uruchamia pełne przetwarzanie (z JSON) w osobnym wątku.

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.
            log_text (tk.Text): Widget logu do wyświetlania komunikatów.
            progress_bar (ttk.Progressbar): Pasek postępu.
            progress_label (tk.Label): Etykieta statusu postępu.
            start_btn (tk.Button): Przycisk startowy, który zostanie zablokowany.
            close_btn (tk.Button): Przycisk do zamknięcia okna.

        Returns:
            None
        """
        start_btn.config(state=tk.DISABLED)
        
        def process_in_thread():
            """
            Cel wątku pomocniczego uruchamianego przez start_processing.
            Wywołuje metodę pełnego przetwarzania (process_all_subjects) przekazując
            bieżące ustawienie use_vision, a po zakończeniu aktualizuje stan przycisku startu.

            Args:
                None

            Returns:
                None
            """
            # ZMIANA: Przekaż aktualną wartość use_vision
            self.process_all_subjects(selected_subjects, log_text, progress_bar, progress_label, self.use_vision.get())
            start_btn.config(state=tk.NORMAL, text="ZAKOŃCZONO", bg="gray")
        
        processing_thread = threading.Thread(target=process_in_thread)
        processing_thread.daemon = True
        processing_thread.start()

    def log_message(self, log_text, message):
        """
        Dodaje wiadomość do widgetu logu w oknie.

        Args:
            log_text (tk.Text): Widget logu.
            message (str): Wiadomość do zapisania.

        Returns:
            None
        """
        try:
            log_text.insert(tk.END, f"{message}\n")
            log_text.see(tk.END)
            log_text.update_idletasks()
        except tk.TclError:
            pass

    def update_progress(self, progress_bar, progress_label, current, total, current_task=""):
        """
        Aktualizuje pasek postępu i etykietę statusu.

        Args:
            progress_bar (ttk.Progressbar): Pasek postępu.
            progress_label (tk.Label): Etykieta statusu.
            current (int): Liczba ukończonych kroków.
            total (int): Łączna liczba kroków.
            current_task (str, optional): Aktualnie wykonywane zadanie.

        Returns:
            None
        """
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
        Normalizuje separatory ścieżek zamieniając backslashes na slashe.

        Args:
            path (str | Path): Ścieżka wejściowa.

        Returns:
            str: Znormalizowana ścieżka.
        """
        # Zamień podwójne backslashe na pojedyncze slashe
        normalized = str(path).replace('\\\\', '/').replace('\\', '/')
        return normalized

    def convert_to_relative_path(self, absolute_path):
        """
        Konwertuje ścieżkę absolutną na ścieżkę względną zaczynającą się od `pgverse`.

        Args:
            absolute_path (str | Path): Ścieżka absolutna.

        Returns:
            str: Ścieżka względna lub oryginalna, jeśli `pgverse` nie występuje.
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

    def copy_required_files_to_detekcje(self, ocr_folder, log_text=None):
        """
        Sprawdza i kopiuje wymagane pliki z folderu 'rezultaty' do folderu 'detekcje'.
        Tworzy folder 'detekcje' jeśli nie istnieje.

        Args:
            ocr_folder (Path): Ścieżka do folderu OCR.
            log_text (tk.Text, optional): Widget logu do zapisu komunikatów.

        Returns:
            bool: True, jeśli pliki są gotowe w folderze 'detekcje'.
        """

        try:
            # Define paths
            rezultaty_path = ocr_folder / "rezultaty"
            detekcje_path = ocr_folder / "detekcje"
            ocr_folder_name = ocr_folder.name
            
            # Create detekcje folder if it doesn't exist
            if not detekcje_path.exists():
                detekcje_path.mkdir(parents=True, exist_ok=True)
                if log_text:
                    self.log_message(log_text, f"  📁 Utworzono folder detekcje dla {ocr_folder_name}")
            
            # Check main OCR text file
            main_txt_file_dest = detekcje_path / f"{ocr_folder_name}.txt"
            main_txt_file_source = rezultaty_path / f"{ocr_folder_name}.txt"
            
            # Check latex_wzory.json file
            latex_file_dest = detekcje_path / "latex_wzory.json"
            latex_file_source = rezultaty_path / "wzory" / "latex_wzory.json"
            
            files_copied = False
            
            # Check and copy main OCR text file if needed
            if not main_txt_file_dest.exists() and main_txt_file_source.exists():
                import shutil
                shutil.copy2(main_txt_file_source, main_txt_file_dest)
                if log_text:
                    self.log_message(log_text, f"  📋 Skopiowano {main_txt_file_source.name} do folderu detekcje")
                files_copied = True
            
            # Check and copy latex_wzory.json if needed
            if not latex_file_dest.exists() and latex_file_source.exists():
                import shutil
                shutil.copy2(latex_file_source, latex_file_dest)
                if log_text:
                    self.log_message(log_text, f"  📋 Skopiowano latex_wzory.json do folderu detekcje")
                files_copied = True
            
            # Check if files exist in detekcje folder
            files_ready = main_txt_file_dest.exists()
            
            if log_text:
                if files_copied:
                    self.log_message(log_text, f"  ✅ Pliki zostały skopiowane do folderu detekcje")
                if files_ready:
                    self.log_message(log_text, f"  ✅ Pliki gotowe do przetwarzania w folderze detekcje")
                else:
                    self.log_message(log_text, f"  ⚠️ Brak wymaganych plików w folderze detekcje")
                    if not main_txt_file_source.exists():
                        self.log_message(log_text, f"  ⚠️ Brak pliku {ocr_folder_name}.txt w folderze rezultaty")
            
            return files_ready
        
        except Exception as e:
            if log_text:
                self.log_message(log_text, f"  ❌ Błąd podczas kopiowania plików: {e}")
            return False

    def process_all_subjects(self, selected_subjects, log_text, progress_bar, progress_label, use_vision):
        """
        Przetwarza wszystkie wybrane przedmioty (OCR + Vision opcjonalnie).
        Tworzy pliki JSON, *_chunks.txt i *_base64.txt.

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.
            log_text (tk.Text): Widget logu.
            progress_bar (ttk.Progressbar): Pasek postępu.
            progress_label (tk.Label): Etykieta statusu.
            use_vision (bool): Czy włączyć tryb Vision dla analizy obrazów.

        Returns:
            None
        """
        self.log_message(log_text, "=== ROZPOCZĘCIE PRZETWARZANIA PRZEDMIOTÓW ===")
        self.log_message(log_text, f"🔍 Tryb Vision: {'WŁĄCZONY' if use_vision else 'WYŁĄCZONY'}")
        
        # Inicjalizacja procesorów
        try:
            self.log_message(log_text, "Inicjalizacja procesorów...")
            processor = ImageTextProcessor()
            self.log_message(log_text, "✓ Procesory zainicjalizowane pomyślnie")
        except Exception as e:
            self.log_message(log_text, f"✗ Błąd inicjalizacji procesorów: {e}")
            return
        
        # Liczenie folderów do przetworzenia
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
        files_converted = 0

        if use_vision:
            self.log_message(log_text, "\n🔄 Converting Vision JSON format to standard format...")
            self.convert_vision_json_format(selected_subjects, lambda msg: self.log_message(log_text, msg))
            self.log_message(log_text, f"✅ Vision JSON format conversion complete: {files_converted} entries processed")
        # Przetwarzanie
        for subject_name in selected_subjects:
            self.log_message(log_text, f"\n--- PRZETWARZANIE PRZEDMIOTU: {subject_name} ---")
            
            ocr_folders = subject_ocr_folders[subject_name]
            
            for ocr_folder in ocr_folders:
                processed_count += 1
                self.update_progress(progress_bar, progress_label, processed_count, total_ocr_folders, 
                                f"{subject_name}/{ocr_folder.name}")
                
                self.log_message(log_text, f"\nPrzetwarzanie folderu OCR: {subject_name}/{ocr_folder.name}")
                
                # NOWE: Sprawdź i skopiuj pliki z rezultaty do detekcje jeśli potrzeba
                self.log_message(log_text, f"  🔍 Sprawdzanie plików źródłowych...")
                files_ready = self.copy_required_files_to_detekcje(ocr_folder, log_text)
                
                if not files_ready:
                    self.log_message(log_text, f"  ⚠️ Brak wymaganych plików dla {ocr_folder.name} - pomijam")
                    continue
                
                # Definiuj ścieżkę do folderu detekcje
                detekcje_path = ocr_folder / "detekcje"
                
                # Przetwarzanie tylko pliku txt o nazwie podfolderu
                expected_txt_file = detekcje_path / f"{ocr_folder.name}.txt"
                if not expected_txt_file.exists():
                    self.log_message(log_text, f"  ⚠️ Brak pliku {ocr_folder.name}.txt w {ocr_folder.name}/detekcje")
                    continue
                
                txt_files = [expected_txt_file]  # Lista z jednym plikiem
                
                # Przetwarzanie plików txt

                # Ensure json_data is still available after vision processing
                for txt_file in txt_files:
                    try:
                        self.log_message(log_text, f"  📄 Przetwarzanie pliku: {txt_file.name}")
                        
                        # KROK 1: Przetwórz plik i pobierz texts
                        texts = processor.process_file(str(txt_file))
                        
                        
                        # Dodaj sprawdzenie typu przed użyciem len()
                        if isinstance(texts, (list, tuple, dict, set)):
                            self.log_message(log_text, f"    📋 Znaleziono {len(texts)} elementów do przetworzenia")
                        else:
                            self.log_message(log_text, f"    ⚠ Nieprawidłowy format danych: {type(texts).__name__}")
                            # Utwórz pustą listę aby kontynuować przetwarzanie
                            texts = []
                        
                        # KROK 2: Utwórz JSON z kontekstem obrazów (już przefiltrowany!)
                        if (use_vision):
                            json_data = processor.get_images_with_context_json(texts, selected_subjects, use_vision=True)
                        else:
                            json_data = processor.get_images_with_context_json(texts, selected_subjects, use_vision=False)
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

                            if not json_data and use_vision==False:
                                self.log_message(log_text, f"    ⚠ Brak istniejących obrazów w pliku {txt_file.name}")
                                json_result = False
                            else:
                                self.log_message(log_text, f"    ✓ Znaleziono {len(json_data)} istniejących obrazów")
                                
                                # Zapisz JSON bezpośrednio (bez dodatkowego filtrowania)
                                json_output_file = detekcje_path / f"{txt_file.stem}_filtered_context.json"
                        
                        # DODANE: upewnij się, że json_data to lista, inaczej zamień na pustą
                        if not isinstance(json_data, list):
                            self.log_message(log_text, f"    ⚠ Nieprawidłowy format danych JSON: {type(json_data).__name__}, oczekiwano listy")
                            json_data = []

                        
                            
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
                            if json_result or chunks_result or base64_result:
                                success_count += 1
                            else:
                                error_count += 1
                            
                    except Exception as e:
                        self.log_message(log_text, f"    ✗ Błąd przetwarzania {txt_file.name}: {e}")
                        error_count += 1
        
        # Podsumowanie
        self.log_message(log_text, f"\n=== PODSUMOWANIE PRZETWARZANIA ===")
        self.log_message(log_text, f"🔍 Tryb Vision: {'WŁĄCZONY' if use_vision else 'WYŁĄCZONY'}")
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
            
            # ZMIANA: Usuń zapytanie o graf Neo4j, zostaw tylko informację
            self.log_message(log_text, "\n💡 Przetwarzanie plików zakończone!")
            self.log_message(log_text, "🔗 Aby zarządzać grafem Neo4j, użyj przycisku:")
            self.log_message(log_text, "   'Zarządzanie bazą danych' w głównym oknie")
        else:
            self.log_message(log_text, f"⚠ ZAKOŃCZONO Z {error_count} BŁĘDAMI")
        
        self.update_progress(progress_bar, progress_label, total_ocr_folders, total_ocr_folders, "Zakończono")

    def convert_vision_json_format(self, selected_subjects, log_function=None):
        """
        Konwertuje niepoprawnie sformatowane pliki JSON z trybu Vision 
        na poprawny format łączony.
        Tworzy plik *_context.json w folderze detekcje.

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.
            log_function (Callable, optional): Funkcja logująca komunikaty.

        Returns:
            tuple[int, int]: Liczba przetworzonych folderów i liczba skonwertowanych wpisów JSON.
        """
        if log_function is None:
            log_function = print
            
        log_function("🔄 Starting JSON format conversion for vision data...")
        total_folders_processed = 0
        total_files_converted = 0
        
        for subject_name in selected_subjects:
            log_function(f"\n📚 Processing subject: {subject_name}")
            subject_path = self.subjects_path / subject_name
            
            # Find all OCR folders
            ocr_folders = [item for item in subject_path.iterdir() 
                        if item.is_dir() and item.name != "__pycache__"]
            
            log_function(f"📂 Found {len(ocr_folders)} OCR folders in {subject_name}")
            
            for ocr_folder in ocr_folders:
                ocr_folder_name = ocr_folder.name
                detekcje_path = ocr_folder / "detekcje"
                
                if not detekcje_path.exists():
                    log_function(f"⚠️ Skipping - No 'detekcje' folder in {ocr_folder_name}")
                    continue
                    
                log_function(f"\n🔍 Processing OCR folder: {ocr_folder_name}")
                
                # List of subdirectories to search for JSON files
                subfolders = ["wzory", "tabele", "figury"]
                all_json_data = []
                files_converted = 0
                
                # Check each subfolder for JSON files
                for subfolder_name in subfolders:
                    subfolder_path = detekcje_path / subfolder_name
                    
                    if not subfolder_path.exists():
                        continue
                        
                    json_files = list(subfolder_path.glob("*.json"))
                    log_function(f"  📄 Found {len(json_files)} JSON files in {subfolder_name}")
                    
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                try:
                                    json_content = json.load(f)
                                    
                                    # Process each item in the JSON file
                                    if isinstance(json_content, list):
                                        for item in json_content:
                                            if "relative_path" in item and "description" in item:
                                                # Transform to the correct format
                                                full_path = f"pgverse/rag_codes/subjects/{item['relative_path']}"
                                                transformed_item = {
                                                    full_path: [item["description"]]
                                                }
                                                all_json_data.append(transformed_item)
                                                files_converted += 1
                                            
                                except json.JSONDecodeError:
                                    log_function(f"    ⚠️ Invalid JSON format in {json_file.name}")
                                    
                        except Exception as e:
                            log_function(f"    ❌ Error processing {json_file.name}: {e}")
                
                # Write the combined data to a new JSON file
                if all_json_data:
                    output_json_path = detekcje_path / f"{ocr_folder_name}_context.json"
                    try:
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(all_json_data, f, ensure_ascii=False, indent=2)
                        log_function(f"✅ Created combined JSON file: {output_json_path.name} with {len(all_json_data)} entries")
                        total_files_converted += files_converted
                    except Exception as e:
                        log_function(f"❌ Error creating combined JSON file: {e}")
                else:
                    log_function(f"⚠️ No valid JSON data found for {ocr_folder_name}")
                    
                total_folders_processed += 1
        
        # Summary
        log_function(f"\n📊 SUMMARY:")
        log_function(f"✅ Processed {total_folders_processed} OCR folders")
        log_function(f"📄 Converted {total_files_converted} JSON entries")
        log_function(f"🎉 JSON format conversion complete!")
        
        return total_folders_processed, total_files_converted

    def open_graph_management_window(self, selected_subjects):
        """
        Otwiera okno do zarządzania grafem Neo4j.
        Umożliwia konfigurację połączenia, testowanie, tworzenie relacji i wyświetlanie statystyk.

        Args:
            selected_subjects (list[str]): Lista wybranych przedmiotów.

        Returns:
            None
        """
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Zarządzanie grafem Neo4j")
        graph_window.geometry("1000x900")  # ZWIĘKSZONE z 800 na 900
        graph_window.grab_set()
        
        # DODANE: Ustawienie minimalnego rozmiaru okna
        graph_window.minsize(800, 700)
        
        # Tytuł
        tk.Label(graph_window, text="Zarządzanie grafem Neo4j - Tworzenie relacji między chunkami", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Info o przedmiatch
        tk.Label(graph_window, text=f"Dostępne przedmioty: {', '.join(selected_subjects)}", 
                font=("Arial", 10), wraplength=900).pack(pady=5)
        
        # NOWE: Frame wyboru przedmiotów
        subject_selection_frame = tk.LabelFrame(graph_window, text="Wybór przedmiotów do załadowania", 
                                               font=("Arial", 10, "bold"))
        subject_selection_frame.pack(fill=tk.X, padx=20, pady=5)  # ZMIENIONE pady z 10 na 5
        
        # Checkboxy dla przedmiotów
        subject_vars = {}
        subjects_frame = tk.Frame(subject_selection_frame)
        subjects_frame.pack(fill=tk.X, padx=10, pady=5)  # ZMIENIONE pady z 10 na 5
        
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
        selection_buttons_frame.pack(fill=tk.X, pady=2)  # ZMIENIONE pady z 5 na 2
        
        def select_all_subjects():
            """
            Zaznacza wszystkie checkboxy przedmiotów w panelu wyboru przedmiotów
            w oknie zarządzania grafem (ustawia odpowiadające tk.BooleanVar na True).

            Args:
                None

            Returns:
                None
            """
            for var in subject_vars.values():
                var.set(True)
        
        def deselect_all_subjects():
            """
            Odznacza wszystkie checkboxy przedmiotów w panelu wyboru przedmiotów
            w oknie zarządzania grafem (ustawia odpowiadające tk.BooleanVar na False).

            Args:
                None

            Returns:
                None
            """
            for var in subject_vars.values():
                var.set(False)
        
        tk.Button(selection_buttons_frame, text="Zaznacz wszystkie", 
                 command=select_all_subjects).pack(side=tk.LEFT, padx=5)
        tk.Button(selection_buttons_frame, text="Odznacz wszystkie", 
                 command=deselect_all_subjects).pack(side=tk.LEFT, padx=5)
        
        # Konfiguracja połączenia
        connection_frame = tk.LabelFrame(graph_window, text="Konfiguracja połączenia Neo4j", 
                                        font=("Arial", 10, "bold"))
        connection_frame.pack(fill=tk.X, padx=20, pady=5)  # ZMIENIONE pady z 10 na 5
        
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
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)  # ZMIENIONE pady z 10 na 5
        
        text_frame = tk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        graph_log_text = tk.Text(text_frame, yscrollcommand=scrollbar.set, 
                                font=("Consolas", 9), wrap=tk.WORD, height=12)  # ZMNIEJSZONE z 15 na 12
        graph_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=graph_log_text.yview)
        
        # Zmienne dla komponentów grafu
        neo4j_connector = None
        graph_builder = None
        
        def log_graph_message(message):
            """
            Dodaje wiadomość do widgetu logu w oknie zarządzania grafem.

            Args:
                message (str): Tekst wiadomości do wyświetlenia w logu.

            Returns:
                None
            """
            try:
                graph_log_text.insert(tk.END, f"{message}\n")
                graph_log_text.see(tk.END)
                graph_log_text.update_idletasks()
            except tk.TclError:
                pass
        
        def test_connection():
            """
            Testuje połączenie z bazą Neo4j przy użyciu aktualnych
            danych logowania (URI, użytkownik, hasło).

            Args:
                None

            Returns:
                None
            """
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
            """
            Nawiązuje połączenie z bazą Neo4j i inicjalizuje obiekt GraphBuilder
            do operacji na grafie. Ustawia przyciski operacyjne jako aktywne.

            Args:
                None

            Returns:
                None
            """
            nonlocal neo4j_connector, graph_builder
            
            try:
                log_graph_message("🔄 Łączenie z Neo4j...")
                neo4j_connector = Neo4jConnector(uri_var.get(), user_var.get(), password_var.get())
                
                log_graph_message("🔄 Inicjalizacja komponentów grafu...")
                # ZMIANA: Użyj stałej SIMILARITY_THRESHOLD
                graph_builder = GraphBuilder(neo4j_connector, similarity_threshold=RELATION_SIMILARITY_THRESHOLD)
                log_graph_message(f"📊 Threshold podobieństwa ustawiony na: {RELATION_SIMILARITY_THRESHOLD}")
                
                log_graph_message("✓ Pomyślnie połączono z Neo4j")
                
                for btn in operation_buttons:
                    btn.config(state=tk.NORMAL)
                
                connect_btn.config(state=tk.DISABLED, text="Połączono")
                disconnect_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                log_graph_message(f"✗ Błąd połączenia: {e}")
        
        def disconnect_from_neo4j():
            """
            Rozłącza bieżące połączenie z Neo4j oraz zwalnia zasoby GraphBuilder.
            Przywraca możliwość ponownego połączenia.

            Args:
                None

            Returns:
                None
            """
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
        def graph_relations():
            """
            Tworzy relacje podobieństwa między węzłami grafu w bazie Neo4j.
            Operacja wykonywana jest w osobnym wątku, aby nie blokować UI.
            W logu wyświetlane są informacje o liczbie węzłów i postępie.

            Args:
                None

            Returns:
                None
            """
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return

            def worker():
                """
                Wątek roboczy odpowiedzialny za tworzenie relacji podobieństwa między węzłami
                w grafie Neo4j. Wykonuje zapytania informacyjne (np. liczba węzłów),
                estymuje czas pracy dla dużych zbiorów, uruchamia proces tworzenia relacji
                poprzez GraphBuilder i raportuje postęp do logu (poprzez progress callback).

                Args:
                    None (korzysta z obiektów zewnętrznych: neo4j_connector, graph_builder, log_graph_message)

                Returns:
                    None
                """
                try:
                    log_graph_message("🔄 Rozpoczynam tworzenie relacji podobieństwa...")
                    
                    # Sprawdź liczbę węzłów (tylko informacyjnie)
                    with neo4j_connector.get_driver().session() as session:
                        count_result = session.run("MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as total")
                        total_nodes = count_result.single()["total"]
                        log_graph_message(f"📊 Węzłów z embeddingami: {total_nodes}")
                        
                        if total_nodes > 2000:
                            log_graph_message("⏰ To może potrwać kilka minut dla dużej liczby węzłów")
                            estimated_time = (total_nodes * total_nodes / 2) / 10000
                            log_graph_message(f"🕐 Przybliżony czas: {estimated_time:.1f} minut")
                        
                        log_graph_message("🚀 Rozpoczynam przetwarzanie wszystkich węzłów...")
                    
                    # Użyj wersji z callbackiem - BEZ LIMITÓW
                    def progress_callback(message):
                        """
                        Prosty callback używany podczas tworzenia relacji — przekazuje komunikat
                        dalej do funkcji logującej (log_graph_message).

                        Args:
                            message (str): Tekst komunikatu postępu.

                        Returns:
                            None
                        """
                        log_graph_message(message)
                    
                    graph_builder.create_relations_with_progress_callback(progress_callback)
                    log_graph_message("✓ Relacje podobieństwa utworzone pomyślnie")
                    
                except Exception as e:
                    log_graph_message(f"✗ Błąd tworzenia relacji: {e}")
                    import traceback
                    log_graph_message(f"Traceback: {traceback.format_exc()}")

            threading.Thread(target=worker, daemon=True).start()

        def show_statistics():
            """
            Pobiera i wyświetla szczegółowe statystyki grafu z bazy Neo4j,
            w tym liczbę węzłów, pokrycie embeddingami i rozkład typów węzłów.
            Operacja wykonywana jest w osobnym wątku.

            Args:
                None

            Returns:
                None
            """
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return

            def worker():
                """
                Wątek pomocniczy do pobierania i raportowania statystyk grafu.
                Pobiera różne metryki (liczba węzłów, pokrycie embeddingami, rozkład typów itp.)
                przy użyciu sesji Neo4j / metod GraphBuilder i zapisuje szczegółowy log.

                Args:
                    None (korzysta z obiektów zewnętrznych: neo4j_connector, graph_builder, log_graph_message)

                Returns:
                    None
                """
                try:
                    log_graph_message("🔄 Pobieranie statystyk grafu...")
                    
                    # Pobierz statystyki z analyze_learning_patterns
                    try:
                        stats = graph_builder.analyze_learning_patterns()
                        
                        if stats and isinstance(stats, dict):
                            log_graph_message("=" * 80)
                            log_graph_message("📊 SZCZEGÓŁOWE STATYSTYKI GRAFU")
                            log_graph_message("=" * 80)
                            
                            # === STATYSTYKI OGÓLNE ===
                            if 'total_statistics' in stats:
                                total_stats = stats['total_statistics']
                                log_graph_message("\n🎯 STATYSTYKI OGÓLNE:")
                                log_graph_message(f"📊 Węzłów razem: {total_stats.get('total_nodes', 0):,}")
                                log_graph_message(f"🧮 Węzłów z embeddingami: {total_stats.get('nodes_with_embeddings', 0):,}")
                                log_graph_message(f"📋 Węzłów z base64: {total_stats.get('nodes_with_base64', 0):,}")
                                
                                # Procentowy rozkład
                                total_nodes = total_stats.get('total_nodes', 0)
                                if total_nodes > 0:
                                    emb_percent = (total_stats.get('nodes_with_embeddings', 0) / total_nodes) * 100
                                    b64_percent = (total_stats.get('nodes_with_base64', 0) / total_nodes) * 100
                                    log_graph_message(f"📈 Pokrycie embeddingami: {emb_percent:.1f}%")
                                    log_graph_message(f"📈 Pokrycie base64: {b64_percent:.1f}%")
                            
                            # === STATYSTYKI WĘZŁÓW WEDŁUG TYPU ===
                            if 'node_statistics_by_type' in stats:
                                node_stats = stats['node_statistics_by_type']
                                log_graph_message("\n📋 ROZKŁAD TYPÓW WĘZŁÓW:")
                                
                                total_by_type = sum(stat['count'] for stat in node_stats)
                                for stat in sorted(node_stats, key=lambda x: x['count'], reverse=True):
                                    data_type = stat['data_type']
                                    count = stat['count']
                                    with_base64 = stat.get('with_base64', 0)
                                    sample_sources = stat.get('sample_sources', [])
                                    
                                    # Ikona dla typu
                                    type_icon = {
                                        'text': '📝', 'image': '🖼️', 
                                        'formula': '🧮', 'table': '📊'
                                    }.get(data_type, '📄')
                                    
                                    percentage = (count / total_by_type * 100) if total_by_type > 0 else 0
                                    base64_percentage = (with_base64 / count * 100) if count > 0 else 0
                                    
                                    log_graph_message(f"  {type_icon} {data_type.upper()}:")
                                    log_graph_message(f"    • Węzłów: {count:,} ({percentage:.1f}%)")
                                    log_graph_message(f"    • Z base64: {with_base64:,} ({base64_percentage:.1f}%)")
                                    if sample_sources:
                                        log_graph_message(f"    • Źródła: {', '.join(sample_sources)}")
                            
                            # === STATYSTYKI RELACJI ===
                            if 'relation_statistics' in stats:
                                rel_stats = stats['relation_statistics']
                                log_graph_message("\n🔗 STATYSTYKI RELACJI:")
                                
                                if rel_stats:
                                    total_relations = sum(stat['count'] for stat in rel_stats)
                                    log_graph_message(f"🎯 Relacji razem: {total_relations:,}")
                                    
                                    for stat in sorted(rel_stats, key=lambda x: x['count'], reverse=True):
                                        rel_type = stat['relation_type']
                                        count = stat['count']
                                        avg_weight = stat.get('avg_weight', 0)
                                        min_weight = stat.get('min_weight', 0)
                                        max_weight = stat.get('max_weight', 0)
                                        
                                        log_graph_message(f"  🔗 {rel_type}:")
                                        log_graph_message(f"    • Liczba: {count:,}")
                                        log_graph_message(f"    • Średnia waga: {avg_weight:.3f}")
                                        log_graph_message(f"    • Zakres wag: {min_weight:.3f} - {max_weight:.3f}")
                                        
                                        # Ocena jakości relacji
                                        if avg_weight > 0.95:
                                            quality = "🟢 Bardzo wysokiej jakości"
                                        elif avg_weight > 0.90:
                                            quality = "🟡 Wysokiej jakości"
                                        elif avg_weight > 0.85:
                                            quality = "🟠 Średniej jakości"
                                        else:
                                            quality = "🔴 Niskiej jakości"
                                        
                                        log_graph_message(f"    • Jakość: {quality}")
                                else:
                                    log_graph_message("  📭 Brak relacji w grafie")
                            
                            # === STATYSTYKI ŹRÓDEŁ ===
                            if 'source_statistics' in stats:
                                source_stats = stats['source_statistics']
                                log_graph_message("\n📚 ROZKŁAD ŹRÓDEŁ:")
                                
                                total_by_source = sum(stat['node_count'] for stat in source_stats)
                                for stat in sorted(source_stats, key=lambda x: x['node_count'], reverse=True):
                                    source = stat['source']
                                    node_count = stat['node_count']
                                    data_types = stat.get('data_types', [])
                                    
                                    percentage = (node_count / total_by_source * 100) if total_by_source > 0 else 0
                                    
                                    # Ikona dla źródła
                                    source_icon = {
                                        'książka': '📖', 'blog': '💬', 'wikipedia': '🌐',
                                        'artykuł_naukowy': '📄', 'forum': '💭', 'news': '📰'
                                    }.get(source, '📋')
                                    
                                    log_graph_message(f"  {source_icon} {source.upper()}:")
                                    log_graph_message(f"    • Węzłów: {node_count:,} ({percentage:.1f}%)")
                                    log_graph_message(f"    • Typy danych: {', '.join(data_types)}")
                            
                            # === KONFIGURACJA I WZORCE ===
                            log_graph_message("\n⚙️ KONFIGURACJA GRAFU:")
                            if 'current_threshold' in stats:
                                threshold = stats['current_threshold']
                                log_graph_message(f"🎚️ Próg podobieństwa: {threshold}")
                            
                            if 'usage_patterns_count' in stats:
                                patterns_count = stats['usage_patterns_count']
                                log_graph_message(f"📈 Wzorców użycia: {patterns_count}")
                            

                            # === OCENA ZDROWIA GRAFU ===
                            log_graph_message("\n🏥 OCENA ZDROWIA GRAFU:")
                            
                            # Sprawdź gęstość grafu
                            if 'total_statistics' in stats and 'relation_statistics' in stats and rel_stats:
                                total_nodes = stats['total_statistics'].get('total_nodes', 0)
                                total_relations = sum(stat['count'] for stat in rel_stats)
                                
                                if total_nodes > 1:
                                    max_possible_relations = total_nodes * (total_nodes - 1) / 2
                                    density = total_relations / max_possible_relations
                                    
                                    log_graph_message(f"📊 Gęstość grafu: {density:.4f}")
                                    
                                    if density > 0.1:
                                        log_graph_message("  🔥 Graf bardzo gęsty - może działać wolno")
                                    elif density > 0.01:
                                        log_graph_message("  ⚡ Graf gęsty - dobra łączność")
                                    elif density > 0.001:
                                        log_graph_message("  ✅ Graf optymalny - zrównoważona gęstość")
                                    else:
                                        log_graph_message("  📉 Graf rzadki - rozważ obniżenie progu")
                            
                            # Sprawdź jakość embeddingów
                            if 'total_statistics' in stats:
                                total_stats = stats['total_statistics']
                                total_nodes = total_stats.get('total_nodes', 0)
                                nodes_with_emb = total_stats.get('nodes_with_embeddings', 0)
                                
                                if total_nodes > 0:
                                    emb_coverage = nodes_with_emb / total_nodes
                                    if emb_coverage == 1.0:
                                        log_graph_message("  ✅ Wszystkie węzły mają embeddingi")
                                    elif emb_coverage > 0.9:
                                        log_graph_message("  🟡 Dobne pokrycie embeddingami")
                                    else:
                                        log_graph_message("  🔴 Słabe pokrycie embeddingami - sprawdź dane")
                            
                            log_graph_message("\n✅ Statystyki zaawansowane zakończone pomyślnie")
                            
                        else:
                            log_graph_message("⚠️ Nie udało się pobrać statystyk z analyze_learning_patterns")
                            raise ValueError("Nieprawidłowe dane statystyk")
                            

                    except Exception as e:
                        log_graph_message(f"⚠️ Błąd w statystykach zaawansowanych: {e}")
                        log_graph_message("🔄 Przechodzę do podstawowych statystyk...")
                    
                    # ZAWSZE wykonaj podstawowe statystyki z Neo4j jako backup
                    log_graph_message("\n" + "=" * 80)
                    log_graph_message("📊 PODSTAWOWE STATYSTYKI (BEZPOŚREDNIO Z BAZY)")
                    log_graph_message("=" * 80)
                    
                    try:
                        with neo4j_connector.get_driver().session() as session:
                            # Podstawowe liczby
                            basic_result = session.run("MATCH (n) RETURN count(n) as total_nodes").single()
                            total_nodes = basic_result['total_nodes']
                            
                            emb_result = session.run("MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as nodes_with_embeddings").single()
                            nodes_with_embeddings = emb_result['nodes_with_embeddings']
                            
                            b64_result = session.run("MATCH (n) WHERE n.base64 IS NOT NULL RETURN count(n) as nodes_with_base64").single()
                            nodes_with_base64 = b64_result['nodes_with_base64']
                            
                            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relations").single()
                            total_relations = rel_result['total_relations']
                            
                            log_graph_message(f"📊 Węzłów razem: {total_nodes:,}")
                            log_graph_message(f"🧮 Węzłów z embeddingami: {nodes_with_embeddings:,}")
                            log_graph_message(f"📋 Węzłów z base64: {nodes_with_base64:,}")
                            log_graph_message(f"🔗 Relacji razem: {total_relations:,}")
                            
                            # Typy węzłów
                            if total_nodes > 0:
                                log_graph_message("\n📋 Rozkład typów węzłów:")
                                type_result = session.run("""
                                    MATCH (n) WHERE n.type IS NOT NULL
                                    RETURN n.type as type, count(n) as count
                                    ORDER BY count DESC
                                """)
                                
                                for record in type_result:
                                    log_graph_message(f"  • {record['type']}: {record['count']:,}")
                            
                            # Typy relacji
                            if total_relations > 0:
                                log_graph_message("\n🔗 Rozkład typów relacji:")
                                rel_type_result = session.run("""
                                    MATCH ()-[r]->()
                                    RETURN type(r) as rel_type, count(r) as count,
                                           avg(r.weight) as avg_weight
                                    ORDER BY count DESC
                                """)
                                for record in rel_type_result:
                                    avg_w = record['avg_weight']
                                    avg_str = f", średnia waga: {avg_w:.3f}" if avg_w else ""
                                    log_graph_message(f"  • {record['rel_type']}: {record['count']:,}{avg_str}")
                        
                        log_graph_message("\n✅ Wszystkie statystyki zakończone pomyślnie")
                        
                    except Exception as e:
                        log_graph_message(f"✗ Błąd podstawowych statystyk: {e}")
                
                except Exception as e:
                    log_graph_message(f"✗ Krytyczny błąd pobierania statystyk: {e}")
                    import traceback
                    log_graph_message(f"Traceback: {traceback.format_exc()}")

            # Uruchom w osobnym wątku
            threading.Thread(target=worker, daemon=True).start()
        
        def clear_all_chunks():
            """
            Usuwa wszystkie węzły typu 'chunk' z grafu Neo4j.
            Operacja wymaga potwierdzenia użytkownika i wykonuje się w wątku.

            Args:
                None

            Returns:
                None
            """
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            # POPRAWKA: Użyj graph_window jako parent
            result = messagebox.askyesno("Potwierdzenie", 
                "Czy na pewno chcesz usunąć WSZYSTKIE CHUNKI z grafu?\nTa operacja jest nieodwracalna!",
                parent=graph_window)
            
            if result:
                try:
                    log_graph_message("🔄 Usuwanie wszystkich chunków...")
                    with neo4j_connector.get_driver().session() as session:
                        result = session.run("MATCH (n:TextNode) DETACH DELETE n")
                        deleted_count = result.consume().counters.nodes_deleted
                    log_graph_message(f"✓ Usunięto {deleted_count} chunków tekstowych")
                except Exception as e:
                    log_graph_message(f"✗ Błąd usuwania chunków: {e}")

        def clear_entire_database():
            """
            Usuwa wszystkie węzły i relacje z grafu Neo4j.
            Wymaga podwójnego potwierdzenia użytkownika, aby zapobiec przypadkowemu
            skasowaniu całej bazy. Operacja wykonywana jest w osobnym wątku.

            Args:
                None

            Returns:
                None
            """
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            # POPRAWKA: Użyj graph_window jako parent dla pierwszego potwierdzenia
            first_confirm = messagebox.askyesno("⚠️ UWAGA - NIEBEZPIECZNA OPERACJA", 
                "Czy na pewno chcesz usunąć CAŁĄ BAZĘ DANYCH?\n\n"
                "Ta operacja:\n"
                "• Usunie WSZYSTKIE węzły (Chunk, TextNode, itp.)\n"
                "• Usunie WSZYSTKIE relacje\n"
                "• Wyczyści całą bazę Neo4j\n"
                "• Jest NIEODWRACALNA!\n\n"
                "Czy jesteś absolutnie pewien?",
                parent=graph_window)
            
            if first_confirm:
                # POPRAWKA: Użyj graph_window jako parent dla drugiego potwierdzenia
                confirm = messagebox.askquestion("🚨 OSTATNIE POTWIERDZENIE", 
                    "To jest ostatnia szansa na anulowanie!\n\n"
                    "Kliknij 'yes' aby BEZPOWROTNIE USUNĄĆ całą bazę danych\n"
                    "Kliknij 'no' aby anulować operację",
                    icon='warning',
                    parent=graph_window)
                
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
            """
            Uruchamia procedury konserwacji w grafie Neo4j, takie jak czyszczenie
            osieroconych węzłów, usuwanie duplikatów i optymalizacja struktury.

            Args:
                None

            Returns:
                None
            """
            if not graph_builder:
                log_graph_message("✗ Brak połączenia z grafem")
                return

            def worker():
                """
                Wątek roboczy uruchamiający procedury konserwacji grafu (np. usuwanie duplikatów,
                optymalizacje, czyszczenie osieroconych węzłów). Zawiera logikę retry (ponawianie prób)
                przy problemach z połączeniem i raportuje postęp za pomocą callbacków.

                Args:
                    None (korzysta z obiektów zewnętrznych: neo4j_connector, graph_builder, log_graph_message)

                Returns:
                    None
                """
                # POPRAWKA: Przenieś nonlocal na początek funkcji
                nonlocal neo4j_connector, graph_builder
                
                try:
                    log_graph_message("🔄 Uruchamianie pełnej konserwacji grafu z obsługą błędów połączenia...")
                    
                    # NOWE: Konfiguracja robustnej konserwacji
                    max_retries = 3
                    retry_delay = 5  # sekund
                    
                    for attempt in range(max_retries):
                        try:
                            log_graph_message(f"🔄 Próba {attempt + 1}/{max_retries} konserwacji...")
                            
                            # POPRAWKA: Sprawdź czy sterownik nie jest zamknięty przed użyciem
                            if neo4j_connector is None:
                                log_graph_message("❌ Brak aktywnego połączenia - próbuję ponownie nawiązać...")
                                raise Exception("No active connection")
                            
                            # Sprawdź połączenie przed rozpoczęciem - z obsługą błędu zamkniętego sterownika
                            try:
                                with neo4j_connector.get_driver().session() as test_session:
                                    test_session.run("RETURN 1")
                                    log_graph_message("✅ Połączenie aktywne - rozpoczynam konserwację...")
                            except Exception as conn_error:
                                if "closed" in str(conn_error).lower() or "defunct" in str(conn_error).lower():
                                    log_graph_message("🔍 Wykryto zamknięte połączenie - próbuję ponownie nawiązać...")
                                    raise Exception("Connection closed")
                                else:
                                    raise conn_error
                            
                            # Uruchom konserwację z callbackiem
                            def maintenance_callback(message):
                                """
                                Callback wykorzystywany przez procedury konserwacji do raportowania postępu.
                                Przekazuje otrzymany komunikat do centralnej funkcji logującej.

                                Args:
                                    message (str): Tekst komunikatu postępu konserwacji.

                                Returns:
                                    None
                                """

                            
                                log_graph_message(f"  {message}")
                            
                            # POPRAWIONE: Użyj metody z callbackiem jeśli istnieje
                            if hasattr(graph_builder, 'run_maintenance_with_callback'):
                                graph_builder.run_maintenance_with_progress_callback(maintenance_callback)
                            else:
                                graph_builder.run_maintenance()
                            

                            log_graph_message("✅ Konserwacja grafu zakończona pomyślnie")
                            return  # Sukces - wyjdź z pętli
                            
                        except Exception as e:
                            error_msg = str(e)
                            log_graph_message(f"❌ Błąd w próbie {attempt + 1}: {error_msg}")
                            
                            # Sprawdź czy to błąd połączenia
                            if any(keyword in error_msg.lower() for keyword in 
                                   ['connection', 'defunct', 'sessionexpired', 'no data', 'timeout', 'closed']):
                                
                                log_graph_message(f"🔍 Wykryto błąd połączenia - spróbuję ponowić...")
                                
                                if attempt < max_retries - 1:  # Nie ostatnia próba
                                    log_graph_message(f"⏳ Czekam {retry_delay} sekund przed ponowną próbą...")
                                    time.sleep(retry_delay)
                                    
                                    # POPRAWKA: Lepsze odtworzenie połączenia
                                    try:
                                        log_graph_message("🔄 Próba ponownego nawiązania połączenia...")
                                        
                                        # Zamknij stare połączenie bezpiecznie
                                        if neo4j_connector:
                                            try:
                                                neo4j_connector.close()
                                            except Exception:
                                                pass  # Ignoruj błędy zamykania
                                        
                                        # Wyczyść referencje
                                        neo4j_connector = None
                                        graph_builder = None
                                        
                                        # Czekaj chwilę
                                        time.sleep(2)
                                        
                                        # Utwórz nowe połączenie
                                        log_graph_message("🔌 Tworzenie nowego połączenia...")
                                        new_connector = Neo4jConnector(uri_var.get(), user_var.get(), password_var.get())
                                        
                                        # Test nowego połączenia
                                        with new_connector.get_driver().session() as session:
                                            session.run("RETURN 1")
                                        
                                        # Zaktualizuj referencje
                                        neo4j_connector = new_connector
                                        graph_builder = GraphBuilder(neo4j_connector, similarity_threshold=RELATION_SIMILARITY_THRESHOLD)
                                        
                                        log_graph_message("✅ Połączenie ponownie nawiązane")
                                        
                                    except Exception as reconnect_error:
                                        log_graph_message(f"❌ Błąd ponownego połączenia: {reconnect_error}")
                                        continue
                            else:
                                # Inny błąd - nie próbuj ponownie
                                log_graph_message(f"💥 Nieodwracalny błąd: {error_msg}")
                                raise
                    
                    log_graph_message("❌ Wszystkie próby konserwacji nie powiodły się")
                    
                except Exception as e:
                    log_graph_message(f"✗ Krytyczny błąd konserwacji grafu: {e}")
                    import traceback
                    log_graph_message(f"Traceback: {traceback.format_exc()}")
                    
                    # DODATKOWE: Porada dla użytkownika
                    log_graph_message("\n💡 PORADA ROZWIĄZANIA:")
                    log_graph_message("1. Sprawdź stabilność połączenia internetowego")
                    log_graph_message("2. Spróbuj ponownie za kilka minut")
                    log_graph_message("3. Rozważ wykonanie konserwacji w mniejszych częściach:")
                    log_graph_message("   - Użyj 'Utwórz relacje w grafie' zamiast pełnej konserwacji")
                    log_graph_message("   - Lub podziel dane na mniejsze grupy")

            threading.Thread(target=worker, daemon=True).start()

        def clear_all_relations():
            """
            Usuwa wszystkie relacje pomiędzy węzłami w bazie Neo4j.
            Operacja wykonywana jest w osobnym wątku, aby nie blokować interfejsu użytkownika.
            W logu wyświetlane są komunikaty o postępie i ewentualnych błędach.

            Args:
                None

            Returns:
                None
            """
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return
            
            # POPRAWKA: Użyj graph_window jako parent
            confirm = messagebox.askyesno("Potwierdzenie usunięcia relacji", 
                "Czy na pewno chcesz usunąć WSZYSTKIE RELACJE z grafu?\n\n"
                "Ta operacja:\n"
                "• Usunie WSZYSTKIE relacje SIMILAR_TO\n"
                "• Usunie WSZYSTKIE relacje RELATES_TO\n"
                "• Usunie WSZYSTKIE inne relacje\n"
                "• Pozostawi węzły nienaruszone\n"
                "• Jest NIEODWRACALNA!\n\n"
                "Węzły pozostaną, ale będą całkowicie niepowiązane.",
                parent=graph_window)
            
            if confirm:
                def worker():
                    """
                    Wątek usuwający wszystkie relacje z bazy Neo4j w partiach (batch deletion).
                    Funkcja wykonuje potwierdzenie od użytkownika, usuwa relacje w pętlach,
                    raportuje liczbę usuniętych relacji i weryfikuje rezultat.

                    Args:
                        None (korzysta z obiektów zewnętrznych: neo4j_connector, log_graph_message, graph_window)

                    Returns:
                        None
                    """
                    try:
                        log_graph_message("🔄 Rozpoczynam usuwanie WSZYSTKICH relacji...")
                        
                        with neo4j_connector.get_driver().session() as session:
                            # Najpierw policz relacje
                            count_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relations")
                            total_relations = count_result.single()["total_relations"]
                            
                            log_graph_message(f"📊 Znaleziono {total_relations} relacji do usunięcia...")
                            
                            if total_relations == 0:
                                log_graph_message("✅ Brak relacji do usunięcia - graf już oczyszczony")
                                return
                            
                            # Usuń wszystkie relacje (w partiach dla bezpieczeństwa)
                            log_graph_message("🔥 Usuwanie relacji w partiach...")
                            
                            deleted_total = 0
                            batch_size = 10000
                            
                            while True:
                                # Usuń partię relacji
                                delete_result = session.run(f"""
                                    MATCH ()-[r]->()
                                    WITH r LIMIT {batch_size}
                                    DELETE r
                                    RETURN count(r) as deleted
                                """)
                                
                                deleted_batch = delete_result.single()["deleted"]
                                deleted_total += deleted_batch
                                
                                if deleted_batch == 0:
                                    break
                                
                                log_graph_message(f"  🗑️ Usunięto partię: {deleted_batch} relacji (łącznie: {deleted_total})")
                                
                                # Krótka przerwa dla bazy danych
                                time.sleep(0.1)
                            
                            # Weryfikacja
                            verify_result = session.run("MATCH ()-[r]->() RETURN count(r) as remaining")
                            remaining = verify_result.single()["remaining"]
                            
                            if remaining == 0:
                                log_graph_message("✅ WSZYSTKIE RELACJE ZOSTAŁY USUNIĘTE")
                                log_graph_message(f"📊 Usunięto łącznie: {deleted_total} relacji")
                                log_graph_message("🎯 Graf zawiera teraz tylko węzły bez połączeń")
                                log_graph_message("💡 Możesz teraz utworzyć relacje od nowa")
                            else:
                                log_graph_message(f"⚠️ OSTRZEŻENIE: Pozostało {remaining} relacji")
                                log_graph_message("🔄 Spróbuj ponownie lub użyj 'Wyczyść całą bazę'")
                        
                    except Exception as e:
                        log_graph_message(f"💥 BŁĄD podczas usuwania relacji: {e}")
                        import traceback
                        log_graph_message(f"Traceback: {traceback.format_exc()}")

                threading.Thread(target=worker, daemon=True).start()
            else:
                log_graph_message("✅ Operacja usuwania relacji została anulowana")

        def check_graph_status():
            """
            Sprawdza bieżący stan grafu w Neo4j i wyświetla informacje
            o liczbie węzłów, relacji oraz ewentualnych ostrzeżeniach.
            Operacja wykonywana w osobnym wątku.

            Args:
                None

            Returns:
                None
            """
            if not neo4j_connector:
                log_graph_message("✗ Brak połączenia z grafem")
                return

            def worker():
                """
                Wątek sprawdzający szczegółowy status grafu: wykonuje szereg zapytań
                do Neo4j (liczenie węzłów, relacji, pokrycie embeddingami, gęstość grafu itp.)
                i raportuje wyniki w logu. Obsługuje wyjątki oraz wyświetla wskazówki
                co do ewentualnej potrzeby zmian konfiguracji (np. threshold).

                Args:
                    None (korzysta z obiektów zewnętrznych: neo4j_connector, log_graph_message)

                Returns:
                    None
                """
                try:
                    log_graph_message("🔄 Sprawdzanie statusu grafu...")
                    
                    with neo4j_connector.get_driver().session() as session:
                        # POPRAWKA: Oddzielne zapytania dla każdej statystyki
                        log_graph_message("📊 Pobieranie statystyk węzłów...")
                        
                        # 1. Podstawowe liczenie węzłów
                        basic_nodes_result = session.run("MATCH (n) RETURN count(n) as total_nodes").single()
                        total_nodes = basic_nodes_result['total_nodes']
                        
                        # 2. Węzły z embeddingami (POPRAWIONE ZAPYTANIE)
                        embeddings_result = session.run("""
                            MATCH (n) 
                            WHERE n.embedding IS NOT NULL 
                            RETURN count(n) as nodes_with_embeddings
                        """).single()
                        nodes_with_embeddings = embeddings_result['nodes_with_embeddings']
                        
                        # 3. Węzły z base64 (POPRAWIONE ZAPYTANIE)
                        base64_result = session.run("""
                            MATCH (n) 
                            WHERE n.base64 IS NOT NULL 
                            RETURN count(n) as nodes_with_base64
                        """).single()
                        nodes_with_base64 = base64_result['nodes_with_base64']
                        
                        # 4. Relacje
                        log_graph_message("📊 Pobieranie statystyk relacji...")
                        relations_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relations").single()
                        total_relations = relations_result['total_relations']
                        
                        log_graph_message("=== STATUS GRAFU ===")
                        log_graph_message(f"📊 Węzłów razem: {total_nodes}")
                        log_graph_message(f"🧮 Węzłów z embeddingami: {nodes_with_embeddings}")
                        log_graph_message(f"📋 Węzłów z base64: {nodes_with_base64}")
                        log_graph_message(f"🔗 Relacji razem: {total_relations}")
                        
                        # 5. Sprawdź typy węzłów (TYLKO jeśli są węzły)
                        if total_nodes > 0:
                            log_graph_message("\n📋 Analizowanie typów węzłów...")
                            type_result = session.run("""
                                MATCH (n) WHERE n.type IS NOT NULL
                                RETURN n.type as type, count(n) as count
                                ORDER BY count DESC
                            """)
                            
                            log_graph_message("\n📋 Rozkład typów węzłów:")
                            type_found = False
                            for record in type_result:
                                log_graph_message(f"  {record['type']}: {record['count']}")
                                type_found = True
                            
                            if not type_found:
                                log_graph_message("  (brak węzłów z określonym typem)")
                        
                        # 6. Sprawdź typy relacji (TYLKO jeśli są relacje)
                        if total_relations > 0:
                            log_graph_message("\n🔗 Analizowanie typów relacji...")
                            rel_result = session.run("""
                                MATCH ()-[r]->()
                                RETURN type(r) as rel_type, count(r) as count
                                ORDER BY count DESC
                            """)
                            log_graph_message("\n🔗 Rozkład typów relacji:")
                            for record in rel_result:
                                log_graph_message(f"  {record['rel_type']}: {record['count']}")
                        else:
                            log_graph_message("\n🔗 Brak relacji w grafie")
                        
                        # 7. DODATKOWE SPRAWDZENIE - czy nie ma błędnych danych
                        log_graph_message("\n🔍 Weryfikacja poprawności danych...")
                        
                        # Sprawdź czy embeddingi mają sensowną długość
                        embedding_check = session.run("""
                            MATCH (n) WHERE n.embedding IS NOT NULL
                            RETURN min(size(n.embedding)) as min_emb_size, 
                                   max(size(n.embedding)) as max_emb_size,
                                   avg(size(n.embedding)) as avg_emb_size
                            LIMIT 1
                        """).single()
                        
                        if embedding_check and embedding_check['min_emb_size'] is not None:
                            log_graph_message(f"  📏 Rozmiar embeddingów:")
                            log_graph_message(f"    - Minimalny: {embedding_check['min_emb_size']}")
                            log_graph_message(f"    - Maksymalny: {embedding_check['max_emb_size']}")
                            log_graph_message(f"    - Średni: {embedding_check['avg_emb_size']:.1f}")
                            
                            # Sprawdź czy rozmiary są sensowne (powinny być ~512 dla CLIP)
                            if embedding_check['max_emb_size'] > 1000:
                                log_graph_message("  ⚠️ UWAGA: Niektóre embeddingi są podejrzanie duże!")
                            elif embedding_check['min_emb_size'] < 100:
                                log_graph_message("  ⚠️ UWAGA: Niektóre embeddingi są podejrzanie małe!")
                            else:
                                log_graph_message("  ✅ Rozmiary embeddingów wyglądają normalnie")
                        
                        # Sprawdź czy base64 ma sensowną długość
                        if nodes_with_base64 > 0:
                            base64_check = session.run("""
                                MATCH (n) WHERE n.base64 IS NOT NULL
                                RETURN min(size(n.base64)) as min_b64_size, 
                                       max(size(n.base64)) as max_b64_size,
                                       avg(size(n.base64)) as avg_b64_size
                                LIMIT 1
                            """).single()
                            
                            if base64_check and base64_check['min_b64_size'] is not None:
                                log_graph_message(f"  📋 Rozmiar danych base64:")
                                log_graph_message(f"    - Minimalny: {base64_check['min_b64_size']} znaków")
                                log_graph_message(f"    - Maksymalny: {base64_check['max_b64_size']} znaków")
                                log_graph_message(f"    - Średni: {base64_check['avg_b64_size']:.0f} znaków")
                        
                        # KOŃCOWA SANITY CHECK
                        log_graph_message("\n🎯 Podsumowanie zdrowia grafu:")
                        if total_nodes == 0:
                            log_graph_message("  📊 Graf jest pusty - gotowy do załadowania danych")
                        elif nodes_with_embeddings == 0:
                            log_graph_message("  ⚠️ Węzły bez embeddingów - nie można tworzyć relacji!")
                        elif total_relations == 0:
                            log_graph_message("  🔗 Węzły bez relacji - można utworzyć relacje podobieństwa")
                        else:
                            density = (total_relations * 2) / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
                            log_graph_message(f"  📈 Graf aktywny - gęstość relacji: {density:.4f}")
                            if density > 0.1:
                                log_graph_message("  🔥 Graf bardzo gęsty - może działać wolno")
                            elif density < 0.001:
                                log_graph_message("  📉 Graf rzadki - warto sprawdzić threshold podobieństwa")
                            else:
                                log_graph_message("  ✅ Graf ma zrównoważoną gęstość")
                        
                        log_graph_message("\n✅ Sprawdzanie statusu zakończone pomyślnie")
                        
                except Exception as e:
                    log_graph_message(f"✗ Błąd sprawdzania statusu: {e}")
                    import traceback
                    log_graph_message(f"Traceback: {traceback.format_exc()}")

            threading.Thread(target=worker, daemon=True).start()
        # Przyciski
        tk.Button(connection_frame, text="Testuj połączenie", 
                 command=test_connection).grid(row=0, column=4, padx=5, pady=5)
        
        operations_frame = tk.LabelFrame(graph_window, text="Operacje na grafie", 
                                        font=("Arial", 10, "bold"))
        operations_frame.pack(fill=tk.X, padx=20, pady=5, side=tk.BOTTOM)  # DODANE side=tk.BOTTOM
        
        # Przyciski połączenia
        connection_buttons_frame = tk.Frame(operations_frame)
        connection_buttons_frame.pack(fill=tk.X, pady=2)  # ZMIENIONE pady z 5 na 2
        
        connect_btn = tk.Button(connection_buttons_frame, text="Połącz z Neo4j", 
                               command=connect_to_neo4j, bg="green", fg="white", 
                               font=("Arial", 10, "bold"))
        connect_btn.pack(side=tk.LEFT, padx=5)
        
        disconnect_btn = tk.Button(connection_buttons_frame, text="Rozłącz", 
                                  command=disconnect_from_neo4j, bg="red", fg="white", 
                                  font=("Arial", 10, "bold"), state=tk.DISABLED)
        disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        # Przyciski operacji - pierwszy rząd - KOMPAKTOWE
        operation_buttons = []
        
        graph_buttons_frame = tk.Frame(operations_frame)
        graph_buttons_frame.pack(fill=tk.X, pady=2)  # ZMIENIONE pady z 5 na 2
        
        btn1 = tk.Button(graph_buttons_frame, text="ZAŁADUJ DANE DO BAZY", 
                        command=lambda: self.load_selected_subjects_to_neo4j(
                            [subject for subject, var in subject_vars.items() if var.get()],
                            log_graph_message, 
                            neo4j_connector
                        ),
                        state=tk.DISABLED, bg="darkgreen", fg="white", 
                        font=("Arial", 9, "bold"))  # ZMNIEJSZONE font z 10 na 9
        btn1.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn1)
        
        btn2 = tk.Button(graph_buttons_frame, text="Utwórz relacje", 
                        command=graph_relations, state=tk.DISABLED,
                        font=("Arial", 9))  # ZMNIEJSZONE font i text
        btn2.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn2)
        
        btn_status = tk.Button(graph_buttons_frame, text="Status grafu", 
                              command=check_graph_status, state=tk.DISABLED,
                              bg="blue", fg="white", font=("Arial", 9, "bold"))  # ZMNIEJSZONE font
        btn_status.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn_status)
        
        btn3 = tk.Button(graph_buttons_frame, text="Statystyki", 
                        command=show_statistics, state=tk.DISABLED,
                        font=("Arial", 9))  # ZMNIEJSZONE font i text
        btn3.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn3)
        
        # Drugi rząd przycisków - KOMPAKTOWE
        graph_buttons_frame2 = tk.Frame(operations_frame)
        graph_buttons_frame2.pack(fill=tk.X, pady=2)  # ZMIENIONE pady z 5 na 2
        
        btn4 = tk.Button(graph_buttons_frame2, text="Konserwacja", 
                        command=run_maintenance, state=tk.DISABLED,
                        bg="orange", fg="white", font=("Arial", 9, "bold"))  # ZMNIEJSZONE font i text
        btn4.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn4)
        
        btn_load_context = tk.Button(graph_buttons_frame2, text="ZAŁADUJ CHUNKI", 
                                 command=lambda: self.load_chunks_from_context_json(
                                     [subject for subject, var in subject_vars.items() if var.get()],
                                     log_graph_message, 
                                     neo4j_connector
                                 ),
                                 state=tk.DISABLED, bg="purple", fg="white", 
                                 font=("Arial", 9, "bold"))  # ZMNIEJSZONE font i text
        btn_load_context.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn_load_context)
        
        btn_clear_relations = tk.Button(graph_buttons_frame2, text="🗑️ USUŃ RELACJE", 
                                       command=clear_all_relations, state=tk.DISABLED, 
                                       bg="red", fg="white", font=("Arial", 9, "bold"))  # ZMNIEJSZONE font i text
        btn_clear_relations.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn_clear_relations)
        
        btn_clear_chunks = tk.Button(graph_buttons_frame2, text="🗑️ USUŃ CHUNKI", 
                                    command=clear_all_chunks, state=tk.DISABLED, 
                                    bg="darkred", fg="white", font=("Arial", 9, "bold"))  # ZMNIEJSZONE font i text
        btn_clear_chunks.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn_clear_chunks)
        
        btn_clear_all = tk.Button(graph_buttons_frame2, text="💥 WYCZYŚĆ BAZĘ", 
                                 command=clear_entire_database, state=tk.DISABLED, 
                                 bg="darkred", fg="yellow", font=("Arial", 9, "bold"))  # ZMNIEJSZONE font i text
        btn_clear_all.pack(side=tk.LEFT, padx=2, pady=2)
        operation_buttons.append(btn_clear_all)
        
        log_graph_message("=== ZARZĄDZANIE GRAFEM NEO4J ===")
        log_graph_message("1. Testuj połączenie")
        log_graph_message("2. Połącz z Neo4j")
        log_graph_message("3. Użyj przycisków operacji")
        log_graph_message("")
        log_graph_message("⚠️ UWAGA: Operacje usuwania są nieodwracalne!")

    def normalize_image_path(self, image_path):
        """
        Normalizuje podaną ścieżkę obrazu do formatu absolutnego względem katalogu bazowego aplikacji.

        Algorytm usuwa prefiks "pgverse", próbuje odbudować ścieżkę w strukturze `rag_codes/subjects/...`,
        a w przypadku niepowodzenia przeszukuje rekursywnie katalogi "rezultaty".

        Args:
            image_path (str | Path): Ścieżka do obrazu (absolutna lub względna, czasem z prefiksem "pgverse").

        Returns:
            str | None: Znormalizowana ścieżka absolutna do istniejącego pliku albo None, jeśli nie udało się znaleźć pliku.
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

    def load_data_to_neo4j(self, selected_subjects, log_function, neo4j_connector, use_vision = False):
        """
        Ładuje multimodalne dane do bazy Neo4j (obrazy, wzory, tabele, chunki tekstowe, base64).
        Funkcja dodaje węzły (bez tworzenia relacji), generuje embeddingi (obrazy lub tekst zależnie od trybu)
        i raportuje postęp poprzez przekazaną funkcję logującą.

        Args:
            selected_subjects (list[str]): Lista nazw przedmiotów do przetworzenia.
            log_function (Callable[[str], None]): Funkcja do zapisu komunikatów/logów (przyjmuje str).
            neo4j_connector (Neo4jConnector): Obiekt odpowiedzialny za połączenie z Neo4j.
            use_vision (bool, optional): Jeśli True — preferuj embeddingi obrazowe; jeśli False — embeddingi tekstowe. Default False.

        Returns:
            None: Wyniki są zapisywane bezpośrednio w bazie i przez log_function; funkcja nie zwraca wartości.
        """
        if not neo4j_connector:
            log_function("✗ Brak połączenia z Neo4j")
            return
            
        try:
            log_function("🔄 Inicjalizacja komponentów...")
            
            # ZMIANA: Użyj singletona zamiast tworzenia nowej instancji
            embedder = CLIPEmbedder.get_instance()
            log_function("✅ Embedder (singleton) gotowy do użycia")
            
            cuda_available = torch.cuda.is_available()
            log_function(f"🖥️ CUDA dostępne: {cuda_available}")
            
            graph_builder = GraphBuilder(neo4j_connector, similarity_threshold=RELATION_SIMILARITY_THRESHOLD)

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
        Wczytuje pary ścieżka->base64 z pliku `{ocr_folder_name}_base64.txt`.
        Parsuje wzorce w formacie `<image/<ścieżka>/<base64>>`, normalizuje separatory i dopasowuje padding base64.

        Args:
            ocr_folder_name (str): Nazwa folderu OCR (prefix pliku base64).
            detekcje_path (Path): Ścieżka do katalogu "detekcje" dla danego folderu OCR.

        Returns:
            dict[str, str]: Słownik mapujący znormalizowaną ścieżkę obrazu (forward-slash) na string base64.
                            Jeśli plik nie istnieje lub parsowanie się nie powiedzie, zwracany jest pusty słownik.
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
        Znajduje dopasowane dane base64 dla obrazu na podstawie fragmentów ścieżki.

        Najpierw porównuje końcówki ścieżki (folder + nazwa pliku), a jeśli to zawiedzie,
        stosuje fallback do porównania wyłącznie po nazwie pliku lub separatorach.

        Args:
            actual_image_path (str | Path): Rzeczywista ścieżka obrazu na dysku.
            base64_dict (dict[str, str]): Słownik mapujący ścieżki obrazów (string) na dane base64.

        Returns:
            str | None: Ciąg base64 odpowiadający danemu obrazowi albo None, jeśli nie znaleziono dopasowania.
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
        Wyciąga część ścieżki zaczynającą się po katalogu "pgverse".

        Obsługuje zarówno backslash (`\\`), jak i forward slash (`/`).
        Jeśli w ścieżce nie ma "pgverse", zwraca ją w oryginalnej postaci.

        Args:
            path (str): Ścieżka wejściowa (np. "C:/projekty/pgverse/rag_codes/...").

        Returns:
            str: Podścieżka od folderu po "pgverse" lub oryginalna ścieżka, jeśli "pgverse" nie występuje.
        """
        try:
            if 'pgverse' in path:
                # Zamień wszystkie separatory na '/'
                path = path.replace('\\', '/')
                
                parts = path.split('/')
                pgverse_index = -1
                for i, part in enumerate(parts):
                    if part.lower() == 'pgverse':
                        pgverse_index = i
                        break
            
                if pgverse_index >= 0 and pgverse_index < len(parts) - 1:
                    # Zwróć część ścieżki od pgverse do końca
                    return '/'.join(parts[pgverse_index + 1:])
        
            return path
        except Exception:
            return path

    # Dodaj brakujące funkcje pomocnicze
    def determine_data_type_from_path(self, image_path):
        """
        Określa typ danych na podstawie fragmentu ścieżki obrazu.

        Wyszukuje słowa kluczowe z mapowania folderów (np. "wzory", "tabele").
        Jeśli nie pasuje żaden wzorzec, zwraca "image".

        Args:
            image_path (str): Ścieżka do pliku obrazu.

        Returns:
            str: Typ danych - np. "image", "formula", "table".
        """
        path_lower = image_path.lower()
        
        for folder_name, data_type in self.folder_type_mapping.items():
            if folder_name in path_lower:
                return data_type
        
        # Domyślnie image
        return "image"
    
    def get_node_prefix(self, data_type):
        """
        Zwraca prefiks dla ID węzła w grafie na podstawie typu danych.

        Args:
            data_type (str): Typ danych ("image", "formula", "table" lub inny).

        Returns:
            str: Prefiks np. "img", "frm", "tbl", "unk".
        """
        if data_type == "image":
            return "img"
        elif data_type == "formula":
            return "frm"
        elif data_type == "table":
            return "tbl"
        else:
            return "unk"
    
    def find_actual_image_path(self, image_path, detekcje_path):
        """
        Znajduje rzeczywistą ścieżkę pliku obrazu w strukturze folderów.

        Najpierw próbuje dopasować ścieżkę względem katalogu bazowego (usuwając "pgverse"),
        następnie sprawdza w podfolderach `detekcje` (np. figury, wzory, tabele),
        a na końcu szuka pliku bezpośrednio w `detekcje`.

        Args:
            image_path (str): Ścieżka oryginalna do obrazu (np. z JSON).
            detekcje_path (Path): Ścieżka do katalogu "detekcje" dla danego OCR.

        Returns:
            str | None: Absolutna ścieżka do istniejącego pliku obrazu albo None, jeśli nie znaleziono.
        """
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
        """
        Uruchamia proces ładowania wybranych przedmiotów do Neo4j w osobnym wątku.
        Weryfikuje wybór, loguje komunikaty i startuje wątek roboczy korzystający z load_data_to_neo4j.

        Args:
            selected_subjects_for_loading (list[str]): Lista nazw przedmiotów do załadowania.
            log_function (Callable[[str], None]): Funkcja logująca komunikaty.
            neo4j_connector (Neo4jConnector): Obiekt połączenia z Neo4j.

        Returns:
            None: Operacja asynchroniczna — efekty są widoczne w logach i w bazie.
        """
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
            """
            (funkcja zagnieżdżona) Wątek pomocniczy wywoływany przez load_selected_subjects_to_neo4j.
            Wywołuje load_data_to_neo4j i obsługuje logowanie sukcesu lub błędów wraz z tracebackiem.

            Args:
                None (korzysta z zmiennych zamknięcia: selected_subjects_for_loading, log_function, neo4j_connector)

            Returns:
                None
            """
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

    def load_chunks_from_context_json(self, selected_subjects_for_loading, log_function, neo4j_connector):
        """
        Uruchamia asynchroniczne ładowanie chunków kontekstowych (context.json) do Neo4j.
        Sprawdza wybór, startuje wątek pomocniczy i loguje status uruchomienia.

        Args:
            selected_subjects_for_loading (list[str]): Lista nazw przedmiotów do przetworzenia.
            log_function (Callable[[str], None]): Funkcja do zapisu komunikatów/logów.
            neo4j_connector (Neo4jConnector): Obiekt połączenia z Neo4j.

        Returns:
            None
        """
        if not selected_subjects_for_loading:
            log_function("⚠️ Nie wybrano żadnego przedmiotu do załadowania!")
            log_function("📋 Zaznacz przynajmniej jeden przedmiot w sekcji 'Wybór przedmiotów do załadowania'")
            messagebox.showwarning("Brak wyboru przedmiotów", 
                                  "Nie wybrano żadnego przedmiotu do załadowania!\n\n"
                                  "Zaznacz przynajmniej jeden przedmiot w sekcji\n"
                                  "'Wybór przedmiotów do załadowania' przed\n"
                                  "kliknięciem przycisku 'ZAŁADUJ CHUNKI Z KONTEKSTU'.")
            return
        
        log_function(f"🎯 Rozpoczynam ładowanie chunków z kontekstu: {', '.join(selected_subjects_for_loading)}")
        log_function("🔄 Uruchamiam ładowanie w osobnym wątku...")
        
        # Uruchom ładowanie w osobnym wątku
        def load_context_in_thread():
            """
            (funkcja zagnieżdżona) Wątek pomocniczy wywoływany przez load_chunks_from_context_json.
            Wywołuje load_context_data_to_neo4j i loguje zakończenie lub błędy (wraz z tracebackiem).

            Args:
                None (korzysta z zmiennych zamknięcia: selected_subjects_for_loading, log_function, neo4j_connector)

            Returns:
                None
            """
            try:
                self.load_context_data_to_neo4j(selected_subjects_for_loading, log_function, neo4j_connector)
                log_function("🎉 ŁADOWANIE CHUNKÓW Z KONTEKSTU ZAKOŃCZONE")
            except Exception as e:
                log_function(f"💥 BŁĄD KRYTYCZNY podczas ładowania chunków: {e}")
                import traceback
                log_function(f"Traceback: {traceback.format_exc()}")
        
        loading_thread = threading.Thread(target=load_context_in_thread)
        loading_thread.daemon = True
        loading_thread.start()
        
        log_function("✅ Wątek ładowania chunków uruchomiony...")

    def load_context_data_to_neo4j(self, selected_subjects, log_function, neo4j_connector):
        """
        Ładuje chunki kontekstowe z plików context.json do Neo4j.
        Tworzy embeddingi tekstowe (kontekst) i wstawia węzły z base64 (jeżeli dostępne).
        Funkcja wykonuje intensywne operacje IO i korzysta z GraphBuilder.

        Args:
            selected_subjects (list[str]): Lista nazw przedmiotów do przetworzenia.
            log_function (Callable[[str], None]): Funkcja do zapisu logów/komunikatów.
            neo4j_connector (Neo4jConnector): Obiekt połączenia z Neo4j.

        Returns:
            None: Wyniki są zapisywane bezpośrednio w bazie i raportowane log_function.
        """
        if not neo4j_connector:
            log_function("✗ Brak połączenia z Neo4j")
            return
            
        try:
            log_function("🔄 Inicjalizacja komponentów...")
            
            # Użyj singletona embeddera
            embedder = CohereEmbedder.get_instance()
            log_function("✅ Embedder (singleton) gotowy do użycia")
            
            cuda_available = torch.cuda.is_available()
            log_function(f"🖥️ CUDA dostępne: {cuda_available}")
            
            graph_builder = GraphBuilder(neo4j_connector, similarity_threshold=RELATION_SIMILARITY_THRESHOLD)
            log_function("✅ Komponenty zainicjalizowane pomyślnie")
            
            # Statystyki
            total_nodes_added = 0
            total_text_chunks = 0
            total_image_nodes = 0
            total_formula_nodes = 0
            total_table_nodes = 0
            total_errors = 0
            total_context_found = 0
            total_base64_found = 0
            
            log_function("\n=== ŁADOWANIE CHUNKÓW Z KONTEKSTU DO NEO4J ===")
            log_function(f"📋 Przedmioty do przetworzenia: {len(selected_subjects)}")
            log_function("📝 TRYB: Tylko embeddingi tekstowe (kontekst) + base64")
            
            # Policz wszystkie foldery OCR
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
                            
                            # Dodaj każdy chunk jako węzeł tekstowy z embeddingiem TEKSTOWYM
                            for chunk_idx, chunk_text in enumerate(text_chunks):
                                try:
                                    if chunk_idx % 10 == 0:  # Log co 10 chunków
                                        log_function(f"  📝 Przetwarzanie chunku {chunk_idx+1}/{len(text_chunks)}...")
                                    
                                    # Pobierz embedding TEKSTU (nie obrazu)
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

                                                text_embedding = text_embedding.tolist() if hasattr(text_embedding, 'tolist') else text_embedding
                                                # Dodaj węzeł tekstowy z embeddingiem tekstowym
                                                graph_builder.insert_node(
                                                    node_id=unique_node_id,
                                                    data_type="text",
                                                    text=chunk_text,
                                                    embedding=text_embedding,
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
                    
                    # Użyj pliku folderOCR_filtered_context.json (ma identyczną strukturę jak context.json)
                    json_files = list(detekcje_path.glob(f"{subfolder_name}_context.json"))
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
                    
                    # === KROK 4: DODAJ WĘZŁY OBRAZÓW/WZORÓW/TABEL Z EMBEDDINGIEM TEKSTOWYM + BASE64 ===
                    if context_dict:
                        log_function(f"\n🎨 KROK 4: Przetwarzanie {len(context_dict)} obrazów z embeddingami tekstowymi + base64...")
                        
                        image_idx = 0
                        for image_path, context_text in context_dict.items():
                            image_idx += 1
                            try:
                                log_function(f"  🖼️ [{image_idx}/{len(context_dict)}] Przetwarzanie: {Path(image_path).name}")
                                
                                # Określ typ danych na podstawie ścieżki
                                data_type = self.determine_data_type_from_path(image_path)
                                log_function(f"    🏷️ Typ danych: {data_type}")
                                
                                # Znajdź odpowiednie base64 na podstawie ścieżki
                                actual_image_path = self.find_actual_image_path(image_path, detekcje_path)
                                base64_data = None
                                
                                if actual_image_path:
                                    base64_data = self.find_matching_base64(actual_image_path, base64_dict)
                                    if base64_data:
                                        total_base64_found += 1
                                        log_function(f"    📋 Znaleziono dane base64")
                                    else:
                                        log_function(f"    ⚠️ Brak danych base64")
                                else:
                                    log_function(f"    ⚠️ Nie znaleziono fizycznego pliku obrazu")
                                
                                # KLUCZOWA ZMIANA: Użyj kontekstu do embeddingu TEKSTOWEGO
                                if context_text:
                                    log_function(f"    🔄 Generowanie embeddingu tekstowego z kontekstu...")
                                    # Embedding TEKSTU zamiast obrazu
                                    context_embedding = embedder.get_text_embedding(context_text)
                                    
                                    if context_embedding is not None:
                                        log_function(f"    ✅ Embedding tekstowy wygenerowany pomyślnie")
                                        
                                        # Utwórz UNIKALNY ID dla węzła
                                        image_content = f"{subject_name}_{subfolder_name}_{Path(image_path).name}_{data_type}_context"
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
                                                # Konwertuj ścieżkę na względną od pgverse
                                                relative_image_path = self.convert_to_relative_path(image_path)
                                                
                                                log_function(f"    💾 Zapisywanie węzła do bazy...")
                                                # Dodaj węzeł z embeddingiem tekstowym + base64
                                                context_embedding = context_embedding.tolist() if hasattr(context_embedding, 'tolist') else context_embedding
                                                graph_builder.insert_node(
                                                    node_id=unique_image_node_id,
                                                    data_type=data_type,
                                                    text=context_text,  # Tekst kontekstu
                                                    embedding=context_embedding,  # Embedding tekstu
                                                    path=relative_image_path,
                                                    source=source_type,
                                                    base64_data=base64_data  # base64
                                                )
                                                
                                                if data_type == 'image':
                                                    total_image_nodes += 1
                                                elif data_type == 'formula':
                                                    total_formula_nodes += 1
                                                elif data_type == 'table':
                                                    total_table_nodes += 1
                                                    
                                                total_nodes_added += 1
                                                log_function(f"    ✅ Dodano węzeł {data_type} z embeddingiem tekstowym + base64: {Path(image_path).name}")
                                            else:
                                                log_function(f"    ⚠️ Węzeł {data_type} już istnieje w bazie (pomijam)")
                                    else:
                                        log_function(f"    ⚠️ Nie udało się pobrać embeddingu tekstowego")
                                        total_errors += 1
                                else:
                                    log_function(f"    ⚠️ Brak kontekstu tekstowego - pomijam")
                                    total_errors += 1
                                        
                            except Exception as e:
                                log_function(f"    ✗ Błąd przetwarzania {image_path}: {e}")
                                total_errors += 1
                    else:
                        log_function(f"⚠️ KROK 4 POMINIĘTY: Brak danych kontekstu obrazów")
                    
                    # Podsumowanie folderu OCR
                    log_function(f"\n📊 PODSUMOWANIE FOLDERU {subfolder_name}:")
                    log_function(f"  📝 Chunki tekstowe: {text_chunks_loaded}")
                    log_function(f"  🖼️ Obrazy z kontekstem: {len(context_dict) if context_dict else 0}")
                    log_function(f"  📋 Base64 znalezione: {len(base64_dict)}")
                    log_function(f"  🧮 Typ embeddingów: TYLKO TEKSTOWE")
            
            # === PODSUMOWANIE KOŃCOWE ===
            log_function(f"\n{'='*80}")
            log_function(f"🎉 PODSUMOWANIE ŁADOWANIA CHUNKÓW Z KONTEKSTU")
            log_function(f"{'='*80}")
            log_function(f"📝 Kontekst znaleziony dla: {total_context_found} obrazów")
            log_function(f"📋 Dane base64 znalezione: {total_base64_found}")
            log_function(f"🧮 Typ embeddingów: TYLKO TEKSTOWE (nie obrazowe)")
            log_function(f"")
            log_function(f"=== WĘZŁY DODANE DO BAZY ===")
            log_function(f"🖼️ Węzłów obrazów (z kontekstem): {total_image_nodes}")
            log_function(f"🧮 Węzłów wzorów (z kontekstem): {total_formula_nodes}")
            log_function(f"📊 Węzłów tabel (z kontekstem): {total_table_nodes}")
            log_function(f"📝 Węzłów tekstowych: {total_text_chunks}")
            log_function(f"🎯 RAZEM węzłów: {total_nodes_added}")
            log_function(f"⚠️ Błędy: {total_errors}")
            
            log_function(f"\n📋 UWAGA: Węzły zostały dodane BEZ relacji.")
            log_function(f"🔗 Aby utworzyć relacje podobieństwa między węzłami:")
            log_function(f"   1. Użyj przycisku 'Utwórz relacje w grafie'")
            log_function(f"   2. Lub użyj przycisku 'Konserwacja grafu' (pełna konserwacja)")
            
            if total_nodes_added > 0:
                log_function("\n🎉 ŁADOWANIE CHUNKÓW Z KONTEKSTU ZAKOŃCZONE POMYŚLNIE!")
                log_function(f"📈 Graf zawiera teraz:")
                log_function(f"  • {total_text_chunks} węzłów tekstowych (embedding tekstu)")
                log_function(f"  • {total_image_nodes} węzłów obrazów (embedding kontekstu)")
                log_function(f"  • {total_formula_nodes} węzłów wzorów (embedding kontekstu)")
                log_function(f"  • {total_table_nodes} węzłów tabel (embedding kontekstu)")
                log_function(f"  • {total_base64_found} węzłów z danymi base64")
                log_function(f"  • BRAK relacji - dodaj je osobno!")
            else:
                log_function("⚠️ Nie dodano żadnych danych - sprawdź logi błędów")
                
        except Exception as e:
            log_function(f"✗ Krytyczny błąd ładowania chunków z kontekstu: {e}")
            import traceback
            log_function(f"Traceback: {traceback.format_exc()}")

    def run(self):
        """
        Uruchamia pętlę głównego GUI aplikacji (Tkinter).

        Args:
            None

        Returns:
            None
        """
        self.root.mainloop()

if __name__ == "__main__":
    app = SubjectSelectorApp()
    app.run()