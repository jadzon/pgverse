# Scalanie OCR

W folderze `scalanie_ocr` znajdują się rozwiązania dotyczące połączenia wyników detekcji poszczególnych elementów pliku wejściowego.
# 
## Pobierz model

[Kliknij tutaj, aby pobrać model_final.pth](https://drive.google.com/drive/folders/19IaPp07gUZoRvYSguwfnpdY-5etiFplt)

Połączono detekcję elementów graficznych, tabel i wzorów wraz z OCR tekstu.   
Ten projekt zawiera narzędzia do ekstrakcji tekstu, tabel i wzorów z plików PDF/obrazów za pomocą OCR i modeli detekcji. 
Dzięki konteneryzacji w Dockerze można uruchomić cały pipeline OCR w spójnym środowisku na dowolnej maszynie (lokalnie lub w chmurze).

 utworzone zostana  podfoldery:

results/

kx_przetwarzanie/

figures/ ← wycięte figury

tabele/ ← wycięte tabele

wzory/ ← wycięte wzory

kx_page1_result.png ← obrazki z wklejonym białym tłem
#

Następnie skrypt przeczyta wszystkie *_result.png w każdym kX_przetwarzanie/, wykona OCR, a w każdym folderze kX_przetwarzanie/ zapisze plik: kX.txt
#
Celem scalania jest zebranie wykrytych fragmentów (np. pól tekstowych, tabel, wzorów itp.) w spójną strukturę.


