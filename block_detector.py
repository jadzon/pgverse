from typing import List, Dict, Optional
import numpy as np

class BlockGridHelper:
    """Klasa pomocnicza do efektywnego wyszukiwania bloków na schemacie."""
    
    def __init__(self):
        """Inicjalizuje pomocnik do wyszukiwania bloków."""
        self.max_block_distance = 50  # Maksymalna odległość do bloku w pikselach
        self.grid_size = 100  # Domyślny rozmiar siatki
        self.grid = None  # Inicjalizacja siatki
    
    @staticmethod
    def create_block_grid(blocks: List[Dict], grid_size: int) -> List[List[List[Dict]]]:
        """Tworzy siatkę bloków o zadanym rozmiarze."""
        if not blocks:
            return []
            
        # Znajdź wymiary obrazu
        max_x = max(block['coords'][2] for block in blocks)
        max_y = max(block['coords'][3] for block in blocks)
        
        # Oblicz wymiary siatki (używając dzielenia całkowitoliczbowego)
        grid_width = int((max_x + grid_size - 1) // grid_size)
        grid_height = int((max_y + grid_size - 1) // grid_size)
        
        # Inicjalizuj siatkę
        grid = [[[] for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Rozmieść bloki w siatce
        for block in blocks:
            x1, y1, x2, y2 = block['coords']
            grid_x1 = int(x1 // grid_size)
            grid_y1 = int(y1 // grid_size)
            grid_x2 = int(x2 // grid_size)
            grid_y2 = int(y2 // grid_size)
            
            # Upewnij się, że indeksy są w zakresie
            grid_x1 = max(0, min(grid_x1, grid_width - 1))
            grid_y1 = max(0, min(grid_y1, grid_height - 1))
            grid_x2 = max(0, min(grid_x2, grid_width - 1))
            grid_y2 = max(0, min(grid_y2, grid_height - 1))
            
            # Dodaj blok do wszystkich komórek siatki, które przecina
            for y in range(grid_y1, grid_y2 + 1):
                for x in range(grid_x1, grid_x2 + 1):
                    grid[y][x].append(block)
        
        return grid
    
    def find_nearest_block_grid(self, x: int, y: int, blocks: List[Dict], 
                              grid: Dict, grid_size: int) -> Optional[int]:
        """Znajdź najbliższy blok używając siatki."""
        # Oblicz indeks siatki dla punktu
        grid_x = x // grid_size
        grid_y = y // grid_size
        
        # Sprawdź komórkę siatki i sąsiednie komórki
        min_dist = float('inf')
        nearest_block = None
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (grid_x + dx, grid_y + dy)
                if key in grid:
                    for block_idx in grid[key]:
                        block = blocks[block_idx]
                        bx1, by1, bx2, by2 = map(int, block['coords'])
                        
                        # Oblicz odległość do środka bloku
                        center_x = (bx1 + bx2) // 2
                        center_y = (by1 + by2) // 2
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            nearest_block = block_idx
        
        if min_dist <= self.max_block_distance:
            return nearest_block
        return None
    
    @staticmethod
    def get_blocks_between(start_block: Dict, end_block: Dict, blocks: List[Dict], grid: List[List[List[Dict]]], grid_size: int) -> List[Dict]:
        """Znajduje bloki między dwoma blokami w siatce."""
        if not grid:
            return []
            
        # Pobierz współrzędne bloków
        start_coords = start_block['coords']
        end_coords = end_block['coords']
        
        # Oblicz środki bloków
        start_center = (
            int((start_coords[0] + start_coords[2]) // 2),
            int((start_coords[1] + start_coords[3]) // 2)
        )
        end_center = (
            int((end_coords[0] + end_coords[2]) // 2),
            int((end_coords[1] + end_coords[3]) // 2)
        )
        
        # Oblicz pozycje w siatce
        grid_x1 = int(start_center[0] // grid_size)
        grid_y1 = int(start_center[1] // grid_size)
        grid_x2 = int(end_center[0] // grid_size)
        grid_y2 = int(end_center[1] // grid_size)
        
        # Upewnij się, że indeksy są w zakresie
        grid_x1 = max(0, min(grid_x1, len(grid[0]) - 1))
        grid_y1 = max(0, min(grid_y1, len(grid) - 1))
        grid_x2 = max(0, min(grid_x2, len(grid[0]) - 1))
        grid_y2 = max(0, min(grid_y2, len(grid) - 1))
        
        # Zbierz bloki z komórek siatki między blokami
        blocks_between = []
        min_y = int(min(grid_y1, grid_y2))
        max_y = int(max(grid_y1, grid_y2))
        min_x = int(min(grid_x1, grid_x2))
        max_x = int(max(grid_x1, grid_x2))
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                for block in grid[y][x]:
                    if block not in [start_block, end_block] and block not in blocks_between:
                        blocks_between.append(block)
        
        return blocks_between

    def are_blocks_connected(self, block1: Dict, block2: Dict, grid: List[List[List[Dict]]]) -> bool:
        """Sprawdza czy dwa bloki są połączone na siatce."""
        if not grid:
            return False
            
        # Pobierz pozycje bloków na siatce
        coords1 = block1['coords']
        coords2 = block2['coords']
        
        grid_x1_1 = int(coords1[0] // self.grid_size)
        grid_y1_1 = int(coords1[1] // self.grid_size)
        grid_x2_1 = int(coords1[2] // self.grid_size)
        grid_y2_1 = int(coords1[3] // self.grid_size)
        
        grid_x1_2 = int(coords2[0] // self.grid_size)
        grid_y1_2 = int(coords2[1] // self.grid_size)
        grid_x2_2 = int(coords2[2] // self.grid_size)
        grid_y2_2 = int(coords2[3] // self.grid_size)
        
        # Sprawdź czy bloki są w sąsiednich komórkach siatki
        for y1 in range(grid_y1_1, grid_y2_1 + 1):
            for x1 in range(grid_x1_1, grid_x2_1 + 1):
                # Sprawdź sąsiednie komórki
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        y2 = y1 + dy
                        x2 = x1 + dx
                        
                        # Sprawdź czy sąsiednia komórka jest w zakresie siatki
                        if (0 <= y2 < len(grid) and 
                            0 <= x2 < len(grid[0])):
                            # Sprawdź czy sąsiednia komórka zawiera drugi blok
                            if block2 in grid[y2][x2]:
                                return True
        
        return False
    
    def get_blocks_between(self, block1: Dict, block2: Dict, blocks: List[Dict]) -> List[Dict]:
        """Znajdź bloki znajdujące się między dwoma blokami."""
        if self.grid is None:
            return []
            
        # Pobierz pozycje bloków na siatce
        coords1 = block1['coords']
        coords2 = block2['coords']
        
        grid_x1_1 = int(coords1[0] // self.grid_size)
        grid_y1_1 = int(coords1[1] // self.grid_size)
        grid_x2_1 = int(coords1[2] // self.grid_size)
        grid_y2_1 = int(coords1[3] // self.grid_size)
        
        grid_x1_2 = int(coords2[0] // self.grid_size)
        grid_y1_2 = int(coords2[1] // self.grid_size)
        grid_x2_2 = int(coords2[2] // self.grid_size)
        grid_y2_2 = int(coords2[3] // self.grid_size)
        
        # Znajdź bloki między nimi
        blocks_between = []
        for i, block in enumerate(blocks):
            if block == block1 or block == block2:
                continue
                
            coords = block['coords']
            grid_x1 = int(coords[0] // self.grid_size)
            grid_y1 = int(coords[1] // self.grid_size)
            grid_x2 = int(coords[2] // self.grid_size)
            grid_y2 = int(coords[3] // self.grid_size)
            
            # Sprawdź czy blok jest między blokami
            if (min(grid_x1_1, grid_x1_2) <= grid_x2 and 
                max(grid_x1_1, grid_x1_2) >= grid_x1 and
                min(grid_y1_1, grid_y1_2) <= grid_y2 and 
                max(grid_y1_1, grid_y1_2) >= grid_y1):
                blocks_between.append(block)
        
        return blocks_between 