import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Dict
from PIL import Image, ImageTk
import os
import json
import shutil
import glob
from dataclasses import dataclass

@dataclass
class Block:
    coords: List[int]  # [x1, y1, x2, y2]
    type: str = "rectangle"  # domyślnie prostokąt

class Calibrator:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Kalibracja detektora bloków")
        
        # Zmienne do przechowywania parametrów
        self.selections = []
        self.current_rect = None
        self.start_x = None
        self.start_y = None
        self.drawing = False
        self.current_image_path = None
        self.dataset_images = []
        self.current_image_index = -1
        self.selected_rect = None  # aktualnie wybrany prostokąt do edycji
        self.editing = False  # czy jesteśmy w trybie edycji
        
        # Katalogi z adnotacjami
        self.annotations_dir = {
            'Automatyka': os.path.join('annotations', 'Automatyka'),
            'Elektroniczne': os.path.join('annotations', 'Elektroniczne')
        }
        
        # Parametry optymalne
        self.optimal_params = self.load_optimal_params()
        
        # Przyciski kontrolne
        self.controls_frame = tk.Frame(self.window)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(self.controls_frame, text="Wczytaj dataset", command=self.load_dataset).pack(side=tk.LEFT)
        tk.Button(self.controls_frame, text="Wyczyść", command=self.clear_selections).pack(side=tk.LEFT)
        tk.Button(self.controls_frame, text="Zapisz i następny", command=self.save_and_next).pack(side=tk.LEFT)
        tk.Button(self.controls_frame, text="Usuń zaznaczony", command=self.delete_selected).pack(side=tk.LEFT)
        tk.Button(self.controls_frame, text="Zakończ", command=self.quit_app).pack(side=tk.LEFT)
        
        # Canvas do wyświetlania obrazu
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bindowanie zdarzeń myszy
        self.canvas.bind("<ButtonPress-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)
        self.canvas.bind("<Button-3>", self.select_rect)  # prawy przycisk myszy do wyboru prostokąta
        
        self.image = None
        self.photo = None
        
        # Tworzenie katalogów na adnotacje i oznaczone obrazy
        os.makedirs('annotations', exist_ok=True)
        os.makedirs('Dataset/Automatyka_reczne', exist_ok=True)
        os.makedirs('Dataset/Elektroniczne_reczne', exist_ok=True)
        os.makedirs(self.annotations_dir['Automatyka'], exist_ok=True)
        os.makedirs(self.annotations_dir['Elektroniczne'], exist_ok=True)
    
    def load_dataset(self):
        """Wczytuje wszystkie obrazy z katalogów Dataset/Automatyka i Dataset/Elektroniczne."""
        self.dataset_images = []
        
        # Wczytaj obrazy z kategorii "automatyka"
        automatyka_dir = os.path.join('Dataset', 'Automatyka')
        if os.path.exists(automatyka_dir):
            for file in os.listdir(automatyka_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.dataset_images.append(os.path.join(automatyka_dir, file))
        
        # Wczytaj obrazy z kategorii "elektroniczne"
        elektroniczne_dir = os.path.join('Dataset', 'Elektroniczne')
        if os.path.exists(elektroniczne_dir):
            for file in os.listdir(elektroniczne_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.dataset_images.append(os.path.join(elektroniczne_dir, file))
        
        if not self.dataset_images:
            messagebox.showwarning("Brak obrazów", "Nie znaleziono obrazów w katalogach Dataset!")
            return
        
        # Przejdź do pierwszego obrazu
        self.current_image_index = 0
        self.load_next_image()
    
    def load_next_image(self):
        """Wczytuje następny obraz z datasetu."""
        if self.current_image_index >= len(self.dataset_images):
            messagebox.showinfo("Koniec", "Przetworzono wszystkie obrazy z datasetu!")
            return
        
        # Wyczyść poprzednie zaznaczenia
        self.selections = []
        self.selected_rect = None
        self.canvas.delete("selection")
        
        self.current_image_path = self.dataset_images[self.current_image_index]
        self.image = cv2.imread(self.current_image_path)
        
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width = self.image.shape[:2]
            
            # Skalowanie obrazu, jeśli jest zbyt duży
            max_size = 800
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                self.image = cv2.resize(self.image, (new_width, new_height))
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
            self.canvas.config(width=self.image.shape[1], height=self.image.shape[0])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Wczytaj istniejące adnotacje, jeśli istnieją
            self.load_existing_annotations()
            
            # Określ typ schematu
            schema_type = 'Automatyka' if 'Automatyka' in self.current_image_path else 'Elektroniczne'
            
            # Aktualizuj tytuł okna z informacją o postępie
            total_images = len(self.dataset_images)
            current_image = self.current_image_index + 1
            self.window.title(f"Kalibracja detektora bloków - {schema_type} - Obraz {current_image}/{total_images} - {os.path.basename(self.current_image_path)}")
    
    def load_existing_annotations(self):
        """Wczytuje istniejące adnotacje dla bieżącego obrazu."""
        if not self.current_image_path:
            return
            
        # Określ typ schematu na podstawie ścieżki
        if 'Automatyka' in self.current_image_path:
            schema_type = 'Automatyka'
        elif 'Elektroniczne' in self.current_image_path:
            schema_type = 'Elektroniczne'
        else:
            return
            
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        annotation_path = os.path.join(self.annotations_dir[schema_type], f'{schema_type}_{base_name}.json')
        
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                    self.selections = data['blocks']
                    self.draw_selections()
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można wczytać adnotacji: {e}")
    
    def draw_selections(self):
        """Rysuje zaznaczone bloki na obrazie."""
        self.canvas.delete("selection")
        for i, selection in enumerate(self.selections):
            coords = selection['coords']
            color = 'blue' if i == self.selected_rect else 'red'
            self.canvas.create_rectangle(
                coords[0], coords[1], coords[2], coords[3],
                outline=color, width=2, tags="selection"
            )
    
    def select_rect(self, event):
        """Wybiera prostokąt do edycji."""
        if not self.selections:
            return
            
        # Znajdź najbliższy prostokąt do kliknięcia
        x, y = event.x, event.y
        min_dist = float('inf')
        selected_idx = None
        
        for i, selection in enumerate(self.selections):
            coords = selection['coords']
            x1, y1, x2, y2 = coords
            
            # Oblicz odległość od punktu kliknięcia do środka prostokąta
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                selected_idx = i
        
        if selected_idx is not None:
            self.selected_rect = selected_idx
            self.draw_selections()
    
    def delete_selected(self):
        """Usuwa wybrany prostokąt."""
        if self.selected_rect is not None:
            self.selections.pop(self.selected_rect)
            self.selected_rect = None
            self.draw_selections()
    
    def clear_selections(self):
        """Czyści wszystkie zaznaczenia."""
        self.selections = []
        self.selected_rect = None
        self.canvas.delete("selection")
    
    def start_rect(self, event):
        """Rozpoczyna rysowanie prostokąta."""
        if self.image is None:
            return
            
        # Jeśli kliknięto na istniejący prostokąt, rozpocznij jego edycję
        if self.selected_rect is not None:
            self.editing = True
            self.start_x = event.x
            self.start_y = event.y
            return
            
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
    
    def draw_rect(self, event):
        """Rysuje prostokąt podczas przeciągania myszy."""
        if not (self.drawing or self.editing):
            return
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        if self.editing and self.selected_rect is not None:
            # Edycja istniejącego prostokąta
            coords = self.selections[self.selected_rect]['coords']
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            
            new_coords = [
                coords[0] + dx,
                coords[1] + dy,
                coords[2] + dx,
                coords[3] + dy
            ]
            
            self.current_rect = self.canvas.create_rectangle(
                *new_coords,
                outline='blue', width=2, tags="selection"
            )
        else:
            # Rysowanie nowego prostokąta
            self.current_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='red', width=2, tags="selection"
            )
    
    def end_rect(self, event):
        """Kończy rysowanie prostokąta."""
        if not (self.drawing or self.editing):
            return
        
        if self.editing and self.selected_rect is not None:
            # Aktualizuj pozycję edytowanego prostokąta
            coords = self.selections[self.selected_rect]['coords']
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            
            self.selections[self.selected_rect]['coords'] = [
                coords[0] + dx,
                coords[1] + dy,
                coords[2] + dx,
                coords[3] + dy
            ]
            self.editing = False
        else:
            # Dodaj nowy prostokąt
            if self.current_rect:
                coords = self.canvas.coords(self.current_rect)
                self.selections.append({
                    'coords': coords,
                    'type': 'rectangle'
                })
        
        self.drawing = False
        self.current_rect = None
        self.draw_selections()
    
    def save_and_next(self):
        """Zapisuje adnotacje i przechodzi do następnego obrazu."""
        if not self.current_image_path or not self.selections:
            return
            
        # Określ typ schematu na podstawie ścieżki
        if 'Automatyka' in self.current_image_path:
            schema_type = 'Automatyka'
        elif 'Elektroniczne' in self.current_image_path:
            schema_type = 'Elektroniczne'
        else:
            messagebox.showerror("Błąd", "Nie można określić typu schematu!")
            return
            
        # Przygotuj dane do zapisania
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        annotation_data = {
            'image_path': self.current_image_path,
            'blocks': self.selections,
            'image_size': {
                'width': self.image.shape[1],
                'height': self.image.shape[0]
            }
        }
        
        # Utwórz katalogi, jeśli nie istnieją
        os.makedirs(self.annotations_dir[schema_type], exist_ok=True)
        os.makedirs(f'Dataset/{schema_type}_reczne', exist_ok=True)
        
        # Zapisz adnotacje
        annotation_path = os.path.join(self.annotations_dir[schema_type], f'{schema_type}_{base_name}.json')
        try:
            with open(annotation_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można zapisać adnotacji: {e}")
            return
            
        # Zapisz oznaczony obraz
        marked_image = self.image.copy()
        for selection in self.selections:
            coords = selection['coords']
            cv2.rectangle(marked_image, 
                         (int(coords[0]), int(coords[1])), 
                         (int(coords[2]), int(coords[3])), 
                         (0, 255, 0), 2)
        
        marked_image_path = os.path.join(f'Dataset/{schema_type}_reczne', f'{base_name}_marked.png')
        try:
            cv2.imwrite(marked_image_path, cv2.cvtColor(marked_image, cv2.COLOR_RGB2BGR))
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można zapisać oznaczonego obrazu: {e}")
            return
            
        # Przejdź do następnego obrazu
        self.current_image_index += 1
        self.selections = []
        self.selected_rect = None
        self.load_next_image()
    
    def analyze_existing_annotations(self):
        """Analizuje wszystkie istniejące adnotacje i aktualizuje parametry optymalne."""
        all_params = {
            'Automatyka': [],
            'Elektroniczne': []
        }
        
        # Zbierz parametry z wszystkich plików adnotacji
        for schema_type in ['Automatyka', 'Elektroniczne']:
            annotations_dir = self.annotations_dir[schema_type]
            if not os.path.exists(annotations_dir):
                continue
                
            for filename in os.listdir(annotations_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(annotations_dir, filename), 'r') as f:
                        data = json.load(f)
                        if 'params' in data:
                            all_params[schema_type].append(data['params'])
        
        # Oblicz średnie parametry dla każdego typu schematu
        for schema_type in ['Automatyka', 'Elektroniczne']:
            if all_params[schema_type]:
                params = all_params[schema_type]
                self.optimal_params[schema_type] = {
                    'min_block_ratio': min(p['min_block_ratio'] for p in params),
                    'max_block_ratio': max(p['max_block_ratio'] for p in params),
                    'min_aspect_ratio': min(p['min_aspect_ratio'] for p in params),
                    'max_aspect_ratio': max(p['max_aspect_ratio'] for p in params),
                    'min_fill_ratio': min(p['min_fill_ratio'] for p in params),
                    'min_area': min(p['min_area'] for p in params),
                    'max_area': max(p['max_area'] for p in params),
                    'min_perimeter': min(p['min_perimeter'] for p in params),
                    'max_perimeter': max(p['max_perimeter'] for p in params),
                    'min_solidity': min(p['min_solidity'] for p in params),
                    'max_solidity': max(p['max_solidity'] for p in params),
                    'avg_block_size': sum(p['avg_block_size'] for p in params) / len(params),
                    'avg_aspect_ratio': sum(p['avg_aspect_ratio'] for p in params) / len(params),
                    'avg_fill_ratio': sum(p['avg_fill_ratio'] for p in params) / len(params),
                    'avg_solidity': sum(p['avg_solidity'] for p in params) / len(params)
                }
        
        # Zapisz zaktualizowane parametry optymalne
        with open('optimal_params.json', 'w') as f:
            json.dump(self.optimal_params, f, indent=2)
    
    def quit_app(self):
        """Zamyka aplikację."""
        if messagebox.askokcancel("Zakończenie", "Czy na pewno chcesz zakończyć pracę?"):
            self.window.quit()
    
    def run(self):
        """Uruchamia aplikację."""
        # Wczytaj pierwszy obraz
        self.load_dataset()
        
        # Uruchom główną pętlę aplikacji
        self.window.mainloop()

    def load_optimal_params(self):
        """Wczytuje parametry optymalne z pliku JSON."""
        try:
            with open('optimal_params.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'Automatyka': {
                    'min_block_ratio': 0.03,
                    'max_block_ratio': 0.3,
                    'min_aspect_ratio': 0.4,
                    'max_aspect_ratio': 2.5,
                    'min_fill_ratio': 0.25,
                    'min_area': 100,
                    'max_area': 10000,
                    'min_perimeter': 40,
                    'max_perimeter': 400,
                    'min_solidity': 0.7,
                    'max_solidity': 1.0,
                    'avg_block_size': 0.1,
                    'avg_aspect_ratio': 1.0,
                    'avg_fill_ratio': 0.3,
                    'avg_solidity': 0.85
                },
                'Elektroniczne': {
                    'min_block_ratio': 0.03,
                    'max_block_ratio': 0.3,
                    'min_aspect_ratio': 0.4,
                    'max_aspect_ratio': 2.5,
                    'min_fill_ratio': 0.25,
                    'min_area': 100,
                    'max_area': 10000,
                    'min_perimeter': 40,
                    'max_perimeter': 400,
                    'min_solidity': 0.7,
                    'max_solidity': 1.0,
                    'avg_block_size': 0.1,
                    'avg_aspect_ratio': 1.0,
                    'avg_fill_ratio': 0.3,
                    'avg_solidity': 0.85
                }
            }

if __name__ == "__main__":
    calibrator = Calibrator()
    calibrator.run()

# Przykład użycia detektora:
# detector = BlockDetector()
# detector.train_classifier('annotations')
# 
# # Wczytaj obraz do analizy
# image = cv2.imread('sciezka/do/obrazu.png')
# if image is not None:
#     blocks = detector.detect_blocks(image)
#     print(f"Znaleziono {len(blocks)} bloków") 