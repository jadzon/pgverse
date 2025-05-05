import os
import re

def read_text_file(file_path):
    """Wczytuje zawartość pliku do łańcucha znaków (lub zwraca None w razie błędu)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Nie udało się wczytać pliku '{file_path}': {e}")
        return None

def longest_common_subsequence_len(a, b):
    """
    Zwraca długość najdłuższego wspólnego podciągu znaków w ciągach a i b.
    Podciąg nie musi być ciągły, ale musi zachowywać kolejność znaków.
    Algorytm dynamiczny O(n*m).
    """
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    
    return dp[n][m]

def measure_lcs_similarity_percent(ref_text, test_text):
    """
    1) Usuwa wszystkie białe znaki.
    2) Liczy długość LCS pomiędzy ciągami znaków.
    3) Zwraca wynik w procentach (względem długości tekstu referencyjnego).
    """
    
    ref_clean = re.sub(r"\s+", "", ref_text)
    test_clean = re.sub(r"\s+", "", test_text)
    
    
    lcs_len = longest_common_subsequence_len(ref_clean, test_clean)
    
    
    ref_len = len(ref_clean)
    if ref_len == 0:
        
        return 100.0 if len(test_clean) == 0 else 0.0
    
    return (lcs_len / ref_len) * 100

def main():
    # Zmień na własne ścieżki plików:
    ref_file_path = "d:\\nauka\\baza\\k3.txt"
    test_file_path = "d:\\nauka\\baza\\wynik3.txt"

    if not os.path.isfile(ref_file_path):
        print(f"Plik referencyjny nie istnieje: {ref_file_path}")
        return
    
    if not os.path.isfile(test_file_path):
        print(f"Plik testowy nie istnieje: {test_file_path}")
        return

    ref_text = read_text_file(ref_file_path)
    test_text = read_text_file(test_file_path)
    
    if ref_text is None or test_text is None:
        print("Błąd odczytu plików.")
        return

    similarity = measure_lcs_similarity_percent(ref_text, test_text)
    print(f"LCS similarity: {similarity:.2f}%")

if __name__ == "__main__":
    main()
