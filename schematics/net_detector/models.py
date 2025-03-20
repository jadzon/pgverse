from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Connection:
    """Klasa reprezentująca połączenie między blokami."""
    start_block: int  # indeks bloku początkowego
    end_block: int    # indeks bloku końcowego
    start_point: Tuple[int, int]  # punkt początkowy połączenia
    end_point: Tuple[int, int]    # punkt końcowy połączenia
    is_directed: bool  # czy połączenie jest kierunkowe
    direction: Optional[str] = None  # kierunek strzałki (np. "up", "down", "left", "right") 