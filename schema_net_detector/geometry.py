from typing import Tuple, Dict

class GeometryHelper:
    """Klasa pomocnicza do operacji geometrycznych."""
    
    @staticmethod
    def line_intersects_block(start: Tuple[int, int], end: Tuple[int, int], block: Dict) -> bool:
        """Sprawdza czy linia przecina blok."""
        x1, y1 = start
        x2, y2 = end
        bx1, by1, bx2, by2 = map(int, block['coords'])
        
        # Sprawdź czy którykolwiek z punktów końcowych jest w bloku
        if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2) or (bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
            return True
            
        # Sprawdź przecięcie z bokami bloku
        # Górny bok
        if GeometryHelper.lines_intersect(x1, y1, x2, y2, bx1, by1, bx2, by1):
            return True
        # Prawy bok
        if GeometryHelper.lines_intersect(x1, y1, x2, y2, bx2, by1, bx2, by2):
            return True
        # Dolny bok
        if GeometryHelper.lines_intersect(x1, y1, x2, y2, bx1, by2, bx2, by2):
            return True
        # Lewy bok
        if GeometryHelper.lines_intersect(x1, y1, x2, y2, bx1, by1, bx1, by2):
            return True
            
        # Sprawdź czy linia przechodzi przez wnętrze bloku
        return GeometryHelper.line_intersects_block_interior(x1, y1, x2, y2, bx1, by1, bx2, by2)
    
    @staticmethod
    def line_intersects_block_interior(x1: int, y1: int, x2: int, y2: int, 
                                     bx1: int, by1: int, bx2: int, by2: int) -> bool:
        """Sprawdza czy linia przechodzi przez wnętrze bloku."""
        # Sprawdź czy linia jest pozioma
        if abs(y2 - y1) < 1:
            # Sprawdź czy linia przechodzi przez blok w poziomie
            if by1 <= y1 <= by2:
                # Sprawdź czy linia przechodzi przez wnętrze bloku
                if (x1 <= bx1 and x2 >= bx2) or (x1 >= bx1 and x2 <= bx2):
                    return True
                # Sprawdź czy linia przecina boki bloku
                if (x1 <= bx1 and x2 >= bx1) or (x1 <= bx2 and x2 >= bx2):
                    return True
            return False
            
        # Sprawdź czy linia jest pionowa
        if abs(x2 - x1) < 1:
            # Sprawdź czy linia przechodzi przez blok w pionie
            if bx1 <= x1 <= bx2:
                # Sprawdź czy linia przechodzi przez wnętrze bloku
                if (y1 <= by1 and y2 >= by2) or (y1 >= by1 and y2 <= by2):
                    return True
                # Sprawdź czy linia przecina boki bloku
                if (y1 <= by1 and y2 >= by1) or (y1 <= by2 and y2 >= by2):
                    return True
            return False
            
        # Dla linii ukośnych, sprawdź czy linia przecina wnętrze bloku
        # Oblicz parametry linii
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        
        # Sprawdź czy linia przechodzi przez wnętrze bloku
        # Sprawdź lewy bok
        y_left = a * bx1 + b
        if by1 <= y_left <= by2:
            if (x1 <= bx1 and x2 >= bx1) or (x1 >= bx1 and x2 <= bx1):
                return True
                
        # Sprawdź prawy bok
        y_right = a * bx2 + b
        if by1 <= y_right <= by2:
            if (x1 <= bx2 and x2 >= bx2) or (x1 >= bx2 and x2 <= bx2):
                return True
                
        # Sprawdź górny bok
        x_top = (by1 - b) / a
        if bx1 <= x_top <= bx2:
            if (y1 <= by1 and y2 >= by1) or (y1 >= by1 and y2 <= by1):
                return True
                
        # Sprawdź dolny bok
        x_bottom = (by2 - b) / a
        if bx1 <= x_bottom <= bx2:
            if (y1 <= by2 and y2 >= by2) or (y1 >= by2 and y2 <= by2):
                return True
                
        return False
    
    @staticmethod
    def lines_intersect(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int, y4: int) -> bool:
        """Sprawdza czy dwie linie się przecinają."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw((x1,y1), (x3,y3), (x4,y4)) != ccw((x2,y2), (x3,y3), (x4,y4)) and ccw((x1,y1), (x2,y2), (x3,y3)) != ccw((x1,y1), (x2,y2), (x4,y4)) 