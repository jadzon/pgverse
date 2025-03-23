import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def wczytaj_obrazy(sciezki_folderow):
    """
    Funkcja wczytująca wszystkie obrazy z podanych folderów.
    
    Args:
        sciezki_folderow: Lista ścieżek do folderów z obrazami
        
    Returns:
        Lista krotek (nazwa_pliku, obraz)
    """
    obrazy = []
    
    for folder in sciezki_folderow:
        for rozszerzenie in ['*.jpg', '*.jpeg', '*.png']:
            for plik in Path(folder).glob(rozszerzenie):
                obraz = cv2.imread(str(plik))
                if obraz is not None:
                    obrazy.append((str(plik), obraz))
    
    print(f"Wczytano {len(obrazy)} obrazów")
    return obrazy

def preprocessing_obrazu(obraz, zastosuj_odszumianie=True):
    """
    Funkcja wykonująca preprocessing obrazu dla wykrywania tekstu.
    
    Args:
        obraz: Obraz w formacie OpenCV (numpy array)
        zastosuj_odszumianie: Flaga określająca czy stosować odszumianie (może zaburzać małe napisy)
        
    Returns:
        Przetworzony obraz
    """
    # Konwersja do skali szarości
    szary_oryginalny = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    
    # Kopia do przetwarzania
    szary = szary_oryginalny.copy()
    
    # Umiarkowane wyrównanie histogramu dla małych napisów
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    szary_wyrownany = clahe.apply(szary)
    
    # Mocniejsze odszumianie 
    if zastosuj_odszumianie:
        # Zwiększone parametry odszumiania
        szary_odszumiony = cv2.fastNlMeansDenoising(szary_wyrownany, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        szary_odszumiony = szary_wyrownany
    
    # Binaryzacja z rozmiarem bloku odpowiednim dla małych napisów
    binaryzacja = cv2.adaptiveThreshold(
        szary_odszumiony, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Zamiast algorytmu Canny, użyjemy bardziej czytelnych metod dla małych napisów:
    
    # 1. Operator gradientu Sobel - bardziej czytelny dla małych napisów
    sobelx = cv2.Sobel(szary_odszumiony, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(szary_odszumiony, cv2.CV_64F, 0, 1, ksize=3)
    
    # Łączymy gradienty
    sobel_combined = cv2.magnitude(sobelx, sobely)
    
    # Normalizacja do zakresu 0-255
    sobel_krawedzie = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    
    # 2. Ulepszona binaryzacja dla małych napisów
    # Zastosujemy dodatkowe przetwarzanie morfologiczne, aby poprawić czytelność
    kernel = np.ones((2, 2), np.uint8)
    binaryzacja_ulepszona = cv2.morphologyEx(binaryzacja, cv2.MORPH_CLOSE, kernel)
    
    return {
        "oryginalny": obraz,
        "szary": szary_oryginalny,
        "szary_wyrownany": szary_wyrownany,
        "szary_odszumiony": szary_odszumiony,
        "binaryzacja": binaryzacja,
        "binaryzacja_ulepszona": binaryzacja_ulepszona,
        "krawedzie": sobel_krawedzie  # Zamieniamy Canny na Sobel
    }

def zapisz_przetworzone_obrazy(przetworzone_obrazy, folder_wyjsciowy):
    """
    Zapisuje przetworzone obrazy do określonego folderu.
    
    Args:
        przetworzone_obrazy: Lista krotek (nazwa_pliku, słownik przetworzonych obrazów)
        folder_wyjsciowy: Ścieżka do folderu wyjściowego
    """
    os.makedirs(folder_wyjsciowy, exist_ok=True)
    
    for nazwa_pliku, obrazy in przetworzone_obrazy:
        nazwa_bazowa = os.path.basename(nazwa_pliku)
        nazwa_bez_rozszerzenia = os.path.splitext(nazwa_bazowa)[0]
        
        for typ_obrazu, obraz in obrazy.items():
            if typ_obrazu != "oryginalny":  # Nie zapisujemy oryginału
                sciezka_wyjsciowa = os.path.join(
                    folder_wyjsciowy, 
                    f"{nazwa_bez_rozszerzenia}_{typ_obrazu}.png"
                )
                cv2.imwrite(sciezka_wyjsciowa, obraz)
    
    print(f"Zapisano przetworzone obrazy w folderze {folder_wyjsciowy}")

def wizualizuj_preprocessing(przetworzone_obrazy, liczba_przykladow=2):
    """
    Wizualizuje wyniki preprocessingu dla kilku przykładowych obrazów.
    
    Args:
        przetworzone_obrazy: Lista krotek (nazwa_pliku, słownik przetworzonych obrazów)
        liczba_przykladow: Liczba obrazów do wyświetlenia
    """
    for i, (nazwa_pliku, obrazy) in enumerate(przetworzone_obrazy[:liczba_przykladow]):
        plt.figure(figsize=(18, 12))
        plt.suptitle(f"Preprocessing obrazu: {os.path.basename(nazwa_pliku)}")
        
        plt.subplot(2, 3, 1)
        plt.title("0. Oryginalny")
        plt.imshow(cv2.cvtColor(obrazy["oryginalny"], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("1. Skala szarości")
        plt.imshow(obrazy["szary"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("2. Szarość wyrównana")
        plt.imshow(obrazy["szary_wyrownany"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("3. Szarość odszumiona")
        plt.imshow(obrazy["szary_odszumiony"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("4. Binaryzacja")
        plt.imshow(obrazy["binaryzacja"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.title("5. Krawędzie")
        plt.imshow(obrazy["krawedzie"], cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def interaktywne_przegladanie(przetworzone_obrazy):
    """
    Funkcja umożliwiająca interaktywne przeglądanie wszystkich przetworzonych obrazów.
    
    Args:
        przetworzone_obrazy: Lista krotek (nazwa_pliku, słownik przetworzonych obrazów)
    """
    if not przetworzone_obrazy:
        print("Brak przetworzonych obrazów do wyświetlenia.")
        return
    
    liczba_obrazow = len(przetworzone_obrazy)
    indeks_obrazu = [0]  # Używamy listy, aby można było modyfikować wartość w funkcji callback
    tryb_pelnowymiarowy = [False]  # Flaga określająca, czy jesteśmy w trybie pełnoekranowym
    wybrany_typ_obrazu = [None]  # Typ obrazu wybrany do wyświetlenia na pełnym ekranie
    
    # Tworzenie figury i osi
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.manager.set_window_title('Przeglądanie przetworzonych obrazów')
    
    # Mapowanie pozycji podwykresów na typy obrazów - dopasowane do układu 3x3
    pozycje_obrazow = {
        (0, 0): "oryginalny",
        (0, 1): "szary",
        (0, 2): "szary_wyrownany",
        (1, 0): "szary_odszumiony", 
        (1, 1): "binaryzacja",
        (1, 2): "binaryzacja_ulepszona",
        (2, 0): "krawedzie"
    }
    
    # Mapowanie typów obrazów na tytuły z numeracją kolejności
    tytuly_obrazow = {
        "oryginalny": "0. Oryginalny",
        "szary": "1. Skala szarości",
        "szary_wyrownany": "2. Szarość wyrównana",
        "szary_odszumiony": "3. Szarość odszumiona",
        "binaryzacja": "4. Binaryzacja",
        "binaryzacja_ulepszona": "5. Binaryzacja ulepszona",
        "krawedzie": "6. Krawędzie (Sobel)"
    }
    
    def wyswietl_obraz_pelny_ekran(indeks, typ_obrazu):
        """Wyświetla pojedynczy obraz na pełnym ekranie"""
        plt.clf()
        nazwa_pliku, obrazy = przetworzone_obrazy[indeks]
        typ_tytul = tytuly_obrazow[typ_obrazu]
        
        fig.suptitle(f"Obraz {indeks+1}/{liczba_obrazow}: {os.path.basename(nazwa_pliku)}\n"
                    f"Typ: {typ_tytul}", fontsize=16)
        
        # Wyświetl wybrany obraz na całym ekranie
        if typ_obrazu == "oryginalny":
            plt.imshow(cv2.cvtColor(obrazy[typ_obrazu], cv2.COLOR_BGR2RGB))
        else:
            cmap = None if typ_obrazu == "oryginalny" else 'gray'
            plt.imshow(obrazy[typ_obrazu], cmap=cmap)
        
        plt.axis('off')
        
        # Dodajemy informację o nawigacji
        plt.figtext(0.5, 0.01, 
                   "Nawigacja: Strzałka w lewo/prawo - zmiana obrazu, ESC/ENTER - powrót do widoku wszystkich, Q - wyjście", 
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        fig.canvas.draw()
    
    def wyswietl_obraz_normalny(indeks):
        """Wyświetla wszystkie typy przetworzenia obrazu"""
        plt.clf()
        nazwa_pliku, obrazy = przetworzone_obrazy[indeks]
        fig.suptitle(f"Obraz {indeks+1}/{liczba_obrazow}: {os.path.basename(nazwa_pliku)}")
        
        plt.subplot(3, 3, 1)
        plt.title("0. Oryginalny")
        plt.imshow(cv2.cvtColor(obrazy["oryginalny"], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(3, 3, 2)
        plt.title("1. Skala szarości")
        plt.imshow(obrazy["szary"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 3)
        plt.title("2. Szarość wyrównana")
        plt.imshow(obrazy["szary_wyrownany"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 4)
        plt.title("3. Szarość odszumiona")
        plt.imshow(obrazy["szary_odszumiony"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 5)
        plt.title("4. Binaryzacja")
        plt.imshow(obrazy["binaryzacja"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 6)
        plt.title("5. Binaryzacja ulepszona")
        plt.imshow(obrazy["binaryzacja_ulepszona"], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 7)
        plt.title("6. Krawędzie (Sobel)")
        plt.imshow(obrazy["krawedzie"], cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)  # Dodajemy miejsce na przyciski
        
        # Dodajemy informację o nawigacji
        plt.figtext(0.5, 0.01, 
                   "Nawigacja: Strzałka w lewo/prawo - zmiana obrazu, Kliknięcie - pełny ekran, Q - wyjście", 
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        fig.canvas.draw()
    
    def nacisniety_klawisz(event):
        """Obsługuje nawigację klawiaturą"""
        if tryb_pelnowymiarowy[0]:
            # W trybie pełnoekranowym
            if event.key in ['escape', 'enter']:
                # Powrót do widoku wszystkich obrazów
                tryb_pelnowymiarowy[0] = False
                wyswietl_obraz_normalny(indeks_obrazu[0])
            elif event.key == 'right' and indeks_obrazu[0] < liczba_obrazow - 1:
                indeks_obrazu[0] += 1
                wyswietl_obraz_pelny_ekran(indeks_obrazu[0], wybrany_typ_obrazu[0])
            elif event.key == 'left' and indeks_obrazu[0] > 0:
                indeks_obrazu[0] -= 1
                wyswietl_obraz_pelny_ekran(indeks_obrazu[0], wybrany_typ_obrazu[0])
            elif event.key == 'q':
                plt.close(fig)
        else:
            # W normalnym trybie
            if event.key == 'right' and indeks_obrazu[0] < liczba_obrazow - 1:
                indeks_obrazu[0] += 1
                wyswietl_obraz_normalny(indeks_obrazu[0])
            elif event.key == 'left' and indeks_obrazu[0] > 0:
                indeks_obrazu[0] -= 1
                wyswietl_obraz_normalny(indeks_obrazu[0])
            elif event.key == 'q':
                plt.close(fig)
    
    def klikniecie_myszy(event):
        """Obsługuje kliknięcia myszy do wyboru typu obrazu"""
        if event.inaxes is None:
            return  # Kliknięcie poza wykresem
        
        if tryb_pelnowymiarowy[0]:
            # W trybie pełnoekranowym kliknięcie przełącza z powrotem do widoku normalnego
            tryb_pelnowymiarowy[0] = False
            wyswietl_obraz_normalny(indeks_obrazu[0])
        else:
            # Określenie, który podwykres został kliknięty
            for i, ax in enumerate(fig.axes):
                if ax == event.inaxes:
                    pozycja = i  # Indeks wykresu (0-7)
                    
                    # Konwersja indeksu na pozycję wiersz, kolumna w siatce 3x3
                    wiersz = pozycja // 3
                    kolumna = pozycja % 3
                    
                    # Sprawdzamy czy dla tej pozycji mamy zdefiniowany typ obrazu
                    typ_obrazu = pozycje_obrazow.get((wiersz, kolumna), None)
                    if typ_obrazu:
                        tryb_pelnowymiarowy[0] = True
                        wybrany_typ_obrazu[0] = typ_obrazu
                        wyswietl_obraz_pelny_ekran(indeks_obrazu[0], typ_obrazu)
                    break
    
    # Podłączenie funkcji obsługi klawiatury i myszy
    fig.canvas.mpl_connect('key_press_event', nacisniety_klawisz)
    fig.canvas.mpl_connect('button_press_event', klikniecie_myszy)
    
    # Wyświetlenie pierwszego obrazu
    wyswietl_obraz_normalny(indeks_obrazu[0])
    
    plt.show()

def main():
    # Ścieżki do folderów z obrazami
    sciezki_folderow = ["Dataset/Automatyka", "Dataset/Elektroniczne"]
    
    # Folder wyjściowy do zapisania przetworzonych obrazów
    folder_wyjsciowy = "Dataset/Przetworzone"
    
    # Zawsze stosuj odszumianie
    zastosuj_odszumianie = True
    
    print("Odszumianie: włączone")
    
    # Wczytaj obrazy
    obrazy = wczytaj_obrazy(sciezki_folderow)
    
    # Wykonaj preprocessing dla każdego obrazu
    przetworzone_obrazy = []
    for nazwa_pliku, obraz in obrazy:
        przetworzone = preprocessing_obrazu(obraz, zastosuj_odszumianie)
        przetworzone_obrazy.append((nazwa_pliku, przetworzone))
    
    # Zapisz przetworzone obrazy
    zapisz_przetworzone_obrazy(przetworzone_obrazy, folder_wyjsciowy)
    
    # Interaktywne przeglądanie wszystkich przetworzonych obrazów
    interaktywne_przegladanie(przetworzone_obrazy)

if __name__ == "__main__":
    main()