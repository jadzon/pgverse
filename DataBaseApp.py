import tkinter as tk
from pathlib import Path
import json
import tkinter.messagebox
from tkinter import ttk

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
        
        # Aktualizacja scroll region
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
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
        
        # Opcjonalnie: zamknij aplikację lub przejdź do następnego etapu
        result = tk.messagebox.askyesno("Kontynuacja", 
                                       "Konfiguracja została zapisana.\n"
                                       "Czy chcesz zamknąć aplikację?")
        if result:
            self.root.destroy()

    def check_sources_complete(self, subject_name):
        """Sprawdza czy wszystkie źródła są przypisane dla danego przedmiotu"""
        subject_path = self.subjects_path / subject_name
        
        # Pobieranie folderów OCR
        ocr_folders = [item for item in subject_path.iterdir() 
                      if item.is_dir() and item.name != "__pycache__"]
        
        if not ocr_folders:
            return True  # Jeśli brak folderów, uznajemy za uzupełnione
        
        # Sprawdź tymczasową konfigurację lub załadowaną z pliku
        sources_config = self.temp_sources_configs.get(subject_name, 
                                                      self.load_sources_config(subject_path))
        
        # Sprawdzenie czy wszystkie foldery mają jakiekolwiek przypisane źródło
        for folder in ocr_folders:
            if folder.name not in sources_config:
                return False  # Folder nie ma przypisanego źródła
        
        return True
    
    def on_checkbox_click(self, subject_name):
        """Obsługuje kliknięcie w checkbox przedmiotu"""
        var = self.subject_vars[subject_name]
        
        # Jeśli próbuje zaznaczyć
        if var.get():
            # Sprawdź czy źródła są przypisane
            if not self.check_sources_complete(subject_name):
                # Odznacz checkbox
                var.set(False)
                # Pokaż okno zarządzania źródłami
                result = tk.messagebox.askyesno("Przypisz źródła", 
                    f"Przedmiot '{subject_name}' ma nieprzypisane źródła.\n"
                    f"Czy chcesz otworzyć zarządzanie źródłami?")
                if result:
                    self.show_sources_manager(subject_name)
        
        self.update_checkbox_states()
    
    def update_checkbox_states(self):
        """Aktualizuje stan wszystkich checkboxów na podstawie przypisania źródeł"""
        for subject_name, checkbox in self.subject_checkboxes.items():
            if self.check_sources_complete(subject_name):
                # Źródła przypisane - normalny kolor
                checkbox.config(fg="black")
            else:
                # Źródła nieprzypisane - szary kolor i odznaczenie
                checkbox.config(fg="gray")
                self.subject_vars[subject_name].set(False)
        
    def load_subjects(self):
        """Ładuje listę przedmiotów z folderu subjects"""
        try:
            if not self.subjects_path.exists():
                error_label = tk.Label(self.checkbox_frame, 
                                     text=f"Folder subjects nie istnieje: {self.subjects_path}", 
                                     fg="red")
                error_label.pack(pady=10)
                return
            
            # Pobranie wszystkich folderów z subjects
            subjects = [item for item in self.subjects_path.iterdir() 
                       if item.is_dir()]
            
            if not subjects:
                no_subjects_label = tk.Label(self.checkbox_frame, 
                                           text="Brak przedmiotów w folderze subjects", 
                                           fg="orange")
                no_subjects_label.pack(pady=10)
                return
            
            # Sortowanie alfabetyczne
            subjects.sort(key=lambda x: x.name.lower())
            
            # Tworzenie checkboxów dla każdego przedmiotu
            for subject in subjects:
                var = tk.BooleanVar()
                self.subject_vars[subject.name] = var
                
                # Frame dla checkboxa i przycisku
                subject_frame = tk.Frame(self.checkbox_frame)
                subject_frame.pack(fill=tk.X, padx=5, pady=2)
                
                checkbox = tk.Checkbutton(subject_frame, 
                                        text=subject.name, 
                                        variable=var,
                                        anchor="w",
                                        command=lambda s=subject.name: self.on_checkbox_click(s))
                checkbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.subject_checkboxes[subject.name] = checkbox
                
                # Przycisk do zarządzania źródłami
                manage_btn = tk.Button(subject_frame, text="Źródła", 
                                     command=lambda s=subject.name: self.show_sources_manager(s),
                                     font=("Arial", 8), width=8)
                manage_btn.pack(side=tk.RIGHT, padx=5)
            
            # Aktualizacja stanów checkboxów
            self.update_checkbox_states()
                
        except Exception as e:
            error_label = tk.Label(self.checkbox_frame, 
                                 text=f"Błąd podczas ładowania przedmiotów: {str(e)}", 
                                 fg="red")
            error_label.pack(pady=10)
    
    def select_all(self):
        """Zaznacza wszystkie przedmioty (tylko te z przypisanymi źródłami)"""
        for subject_name, var in self.subject_vars.items():
            if self.check_sources_complete(subject_name):
                var.set(True)
    
    def deselect_all(self):
        """Odznacza wszystkie przedmioty"""
        for var in self.subject_vars.values():
            var.set(False)
    
    def load_sources_config(self, subject_path):
        """Ładuje konfigurację źródeł z pliku JSON"""
        config_file = subject_path / "sources_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Błąd podczas ładowania konfiguracji: {e}")
                return {}
        return {}
    
    def save_sources_config(self, subject_path, sources_config):
        """Zapisuje konfigurację źródeł do pliku JSON"""
        config_file = subject_path / "sources_config.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(sources_config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Błąd podczas zapisywania konfiguracji: {e}")
            return False
    
    def show_sources_manager(self, subject_name):
        """Wyświetla okno zarządzania źródłami dla wybranego przedmiotu"""
        subject_path = self.subjects_path / subject_name
        
        # Pobieranie folderów OCR
        ocr_folders = [item for item in subject_path.iterdir() 
                      if item.is_dir() and item.name != "__pycache__"]
        
        if not ocr_folders:
            tk.messagebox.showinfo("Informacja", 
                                 f"Brak folderów OCR w przedmiocie '{subject_name}'")
            return
        
        # Sprawdzenie czy istnieje plik konfiguracyjny
        config_file = subject_path / "sources_config.json"
        config_exists = config_file.exists()
        
        # Ładowanie istniejącej konfiguracji (tymczasowej lub z pliku)
        sources_config = self.temp_sources_configs.get(subject_name, 
                                                      self.load_sources_config(subject_path))
        
        # Tworzenie nowego okna
        sources_window = tk.Toplevel(self.root)
        sources_window.title(f"Zarządzanie źródłami - {subject_name}")
        sources_window.geometry("700x600")  # Zwiększona wysokość dla dodatkowych przycisków
        sources_window.grab_set()  # Modal window
        
        # Tytuł
        title_label = tk.Label(sources_window, 
                              text=f"Przypisz źródła do folderów OCR\nPrzedmiot: {subject_name}", 
                              font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # Status istniejącej konfiguracji
        if config_exists:
            status_frame = tk.Frame(sources_window)
            status_frame.pack(pady=5)
            
            status_label = tk.Label(status_frame, 
                                   text="🔍 Znaleziono istniejący plik konfiguracji źródeł", 
                                   font=("Arial", 10, "bold"), fg="orange")
            status_label.pack()
            
            # Przyciski zarządzania istniejącą konfiguracją
            config_control_frame = tk.Frame(status_frame)
            config_control_frame.pack(pady=5)
            
            def reset_to_defaults():
                """Resetuje wszystkie źródła do 'unknown'"""
                result = tk.messagebox.askyesno("Potwierdzenie", 
                    "Czy na pewno chcesz zresetować wszystkie źródła do 'unknown'?\n"
                    "Ta operacja zastąpi wszystkie istniejące ustawienia.")
                if result:
                    for combo in folder_combos.values():
                        combo.set("unknown")
                    # Usuń z tymczasowej konfiguracji
                    if subject_name in self.temp_sources_configs:
                        del self.temp_sources_configs[subject_name]
                    tk.messagebox.showinfo("Reset", "Wszystkie źródła zostały zresetowane do 'unknown'")
            
            def delete_config_file():
                """Usuwa plik konfiguracyjny"""
                result = tk.messagebox.askyesno("Potwierdzenie usunięcia", 
                    f"Czy na pewno chcesz usunąć plik konfiguracji?\n"
                    f"Ścieżka: {config_file}\n\n"
                    f"Po usunięciu wszystkie źródła zostaną zresetowane do 'unknown'.")
                if result:
                    try:
                        config_file.unlink()  # Usuwa plik
                        # Resetuj wszystkie combobox do 'unknown'
                        for combo in folder_combos.values():
                            combo.set("unknown")
                        # Usuń z tymczasowej konfiguracji
                        if subject_name in self.temp_sources_configs:
                            del self.temp_sources_configs[subject_name]
                        tk.messagebox.showinfo("Usunięto", 
                                             "Plik konfiguracji został usunięty.\n"
                                             "Wszystkie źródła zostały zresetowane do 'unknown'.")
                    except Exception as e:
                        tk.messagebox.showerror("Błąd", 
                                               f"Nie udało się usunąć pliku konfiguracji:\n{str(e)}")
            
            reset_btn = tk.Button(config_control_frame, text="Reset do 'unknown'", 
                                 command=reset_to_defaults,
                                 bg="orange", fg="white", font=("Arial", 9))
            reset_btn.pack(side=tk.LEFT, padx=5)
            
            delete_btn = tk.Button(config_control_frame, text="Usuń plik konfiguracji", 
                                  command=delete_config_file,
                                  bg="darkred", fg="white", font=("Arial", 9))
            delete_btn.pack(side=tk.LEFT, padx=5)
            
            # Informacja o aktualnej konfiguracji
            config_info = self.get_config_summary(sources_config, ocr_folders)
            info_text = tk.Text(status_frame, height=3, width=70, font=("Arial", 8))
            info_text.pack(pady=5)
            info_text.insert(tk.END, config_info)
            info_text.config(state=tk.DISABLED)
        
        # Informacja o wymaganiu
        req_label = tk.Label(sources_window, 
                            text="Wszystkie foldery muszą mieć przypisane jakiekolwiek źródło (w tym 'unknown')", 
                            font=("Arial", 9), fg="blue")
        req_label.pack(pady=5)
        
        # Informacja o zapisywaniu
        save_info_label = tk.Label(sources_window, 
                                  text="Uwaga: Zmiany zostaną zapisane do pliku dopiero po kliknięciu 'Dalej' w głównym oknie", 
                                  font=("Arial", 8), fg="red")
        save_info_label.pack(pady=2)
        
        # Frame z scrollbarem
        main_frame = tk.Frame(sources_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas
        canvas = tk.Canvas(main_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        # Frame dla zawartości
        content_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        
        # Słownik do przechowywania comboboxów
        folder_combos = {}
        
        # Tworzenie wierszy dla każdego folderu
        for i, folder in enumerate(sorted(ocr_folders, key=lambda x: x.name.lower())):
            row_frame = tk.Frame(content_frame)
            row_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Nazwa folderu
            folder_label = tk.Label(row_frame, text=folder.name, 
                                   font=("Arial", 10), width=30, anchor="w")
            folder_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Combobox dla wyboru źródła
            source_combo = ttk.Combobox(row_frame, values=self.source_types, 
                                       state="readonly", width=20)
            source_combo.pack(side=tk.LEFT, padx=5)
            
            # Ustawienie domyślnej wartości
            current_source = sources_config.get(folder.name, "unknown")
            if current_source in self.source_types:
                source_combo.set(current_source)
            else:
                source_combo.set("unknown")
            
            # Wskaźnik czy źródło zostało zmienione
            if config_exists and current_source != "unknown":
                status_indicator = tk.Label(row_frame, text="✓", fg="green", font=("Arial", 12, "bold"))
                status_indicator.pack(side=tk.LEFT, padx=5)
            
            folder_combos[folder.name] = source_combo
        
        # Aktualizacja scroll region
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        content_frame.bind("<Configure>", configure_scroll_region)
        
        # Przyciski kontrolne
        button_frame = tk.Frame(sources_window)
        button_frame.pack(pady=10)
        
        def set_all_source(source_type):
            for combo in folder_combos.values():
                combo.set(source_type)
        
        # Przyciski do szybkiego ustawiania
        quick_buttons_frame = tk.Frame(button_frame)
        quick_buttons_frame.pack(pady=5)
        
        tk.Label(quick_buttons_frame, text="Ustaw wszystkie jako:", 
                font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        for source in ["książka", "wikipedia", "artykuł_naukowy", "unknown"]:
            btn = tk.Button(quick_buttons_frame, text=source, 
                           command=lambda s=source: set_all_source(s),
                           font=("Arial", 8))
            btn.pack(side=tk.LEFT, padx=2)
        
        # Przyciski akcji
        action_frame = tk.Frame(button_frame)
        action_frame.pack(pady=10)
        
        def save_temp_config():
            # Zbieranie danych z comboboxów
            new_config = {}
            missing_sources = []
            
            for folder_name, combo in folder_combos.items():
                source = combo.get()
                if source and source in self.source_types:
                    new_config[folder_name] = source
                else:
                    missing_sources.append(folder_name)
            
            # Sprawdzenie czy wszystkie foldery mają przypisane źródła
            if missing_sources:
                tk.messagebox.showwarning("Ostrzeżenie", 
                    f"Następujące foldery nie mają przypisanych źródeł:\n" + 
                    "\n".join(missing_sources))
                return
            
            # Zapisywanie do tymczasowej konfiguracji
            self.temp_sources_configs[subject_name] = new_config
            
            # Różne komunikaty w zależności od tego czy plik już istniał
            if config_exists:
                tk.messagebox.showinfo("Sukces", 
                                     "Konfiguracja źródeł została zaktualizowana tymczasowo!\n"
                                     "Zmiany zastąpią istniejący plik po kliknięciu 'Dalej' w głównym oknie.")
            else:
                tk.messagebox.showinfo("Sukces", 
                                     "Konfiguracja źródeł została zapisana tymczasowo!\n"
                                     "Aby zapisać na stałe, kliknij 'Dalej' w głównym oknie.")
            
            sources_window.destroy()
            # Aktualizacja stanów checkboxów w głównym oknie
            self.update_checkbox_states()
        
        save_btn = tk.Button(action_frame, text="Zapisz tymczasowo", 
                            command=save_temp_config, 
                            bg="lightblue", font=("Arial", 10, "bold"))
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(action_frame, text="Anuluj", 
                              command=lambda: [sources_window.destroy(), self.update_checkbox_states()],
                              bg="lightcoral", font=("Arial", 10))
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Informacja o liczbie folderów
        info_label = tk.Label(sources_window, 
                             text=f"Znaleziono {len(ocr_folders)} folderów OCR", 
                             font=("Arial", 9), fg="gray")
        info_label.pack(pady=5)
    
    def get_config_summary(self, sources_config, ocr_folders):
        """Generuje podsumowanie aktualnej konfiguracji"""
        if not sources_config:
            return "Brak konfiguracji źródeł."
        
        # Zliczanie typów źródeł
        source_counts = {}
        for folder in ocr_folders:
            source_type = sources_config.get(folder.name, "unknown")
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        summary = "Aktualna konfiguracja źródeł:\n"
        for source_type, count in sorted(source_counts.items()):
            summary += f"• {source_type}: {count} folderów\n"
        
        # Sprawdzenie czy wszystkie foldery mają przypisane źródła
        missing_count = len([f for f in ocr_folders if f.name not in sources_config])
        if missing_count > 0:
            summary += f"• BRAK ŹRÓDŁA: {missing_count} folderów\n"
        
        return summary.strip()
    
    def run(self):
        """Uruchamia aplikację"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SubjectSelectorApp()
    app.run()