# Scalanie OCR

W folderze `scalanie_ocr` znajdują się rozwiązania dotyczące połączenia wyników detekcji poszczególnych elementów pliku wejściowego.
# 

Połączono detekcję elementów graficznych, tabel i wzorów wraz z OCR tekstu.   
W tym samym katalogu, obok `detekcja_elementow.py`, uruchom: python wyodrebniony_tekst.py

 utworzone zostana  podfoldery:

results/
k1_przetwarzanie/
figures/ ← wycięte figury
tabele/ ← wycięte tabele
wzory/ ← wycięte wzory
k1_page1_result.png ← obrazki z wklejonym białym tłem

Następnie skrypt przeczyta wszystkie *_result.png w każdym kX_przetwarzanie/, wykona OCR, a w każdym folderze kX_przetwarzanie/ zapisze plik: kX.txt

Celem scalania jest zebranie wykrytych fragmentów (np. pól tekstowych, tabel, wzorów itp.) w spójną strukturę.


