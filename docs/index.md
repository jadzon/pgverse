

# PGVerse
PGVerse to inteligentyny korepetytor który ma za zadanie pomoc uczniom kierunku Automatyka Cybernetyka i Robotyka w nauce. W swoim arsenale PGVerse posiada narzędzia do analizy wykresów oraz schematów, OCR, wbudowany LLM, polski Bielik, który korzysta z wiedzy zawartej w bazie danych oraz narzędzia do obliczania metryk (sprawdzanie spełnienia założeń projektowych).

[Szybki start](quickstart.md){ .md-button .md-button--primary }
[O projekcie](about.md){ .md-button }

Krótko o możliwościach:
- wykrywanie i interpretacja osi wykresów, ekstrakcja danych
- narzędzie OCR do wygrywania własnych materiałów do bazy danych
- Bielik LLM, który przy odpowiedziach posługuje się danymi przykładowymi oraz wgranymi przez użytkownika
- analityka schematów technicznych

## Przegląd sekcji

<div class="grid cards" markdown>

-   :material-poll: **Analiza wykresów (Charts)**  
	Moduły do pracy z wykresami — od przygotowania obrazów po eksport wyników.  
	[Dokumentacja »](reference/charts/charts.md)

-   :material-target: **Przecięcia osi**  
	Wykryj główną linię i pionowe przecięcia na wykresie.  
	[Zobacz moduł »](reference/charts/axis_intersection_detector/main.md)

-   :material-chart-bell-curve-cumulative: **Wykrywanie i interpretacja osi**  
	Grupowanie ramek, interpretacja podpisów, raport trafności.  
	[Przegląd modułów »](reference/charts/charts_axes_detect/axes_detection.md)

-   :material-function-variant: **Analiza danych (regresja symboliczna)**  
	Dopasowanie formuł (PySR), metryki i LaTeX.  
	[Symbolic Regressor »](reference/charts/data_analyzer/symbolic_regressor.md)

-   :material-file-find: **Odczyt danych z wykresu**  
	Wycinanie wykresu, skeletonizacja, ekstrakcja punktów.  
	[Dokumentacja »](reference/charts/read_chart/read_chart.md)

-   :material-text-recognition: **OCR**  
	Pipeline’y do dokumentów i obrazów (EasyOCR, PaddleOCR) oraz narzędzia pomocnicze.  
	[Moduły OCR »](reference/ocr/scalanie_ocr/detekcja_elementow.md)

-   :material-robot: **Metryki RAG**  
	Ocena jakości odpowiedzi, m.in. BERTScore i wizualizacje.  
	[Dokumentacja »](reference/rag_codes/rag_metrics/BERTscore.md)

-   :material-circuit-integrated: **Schematy**  
	Przetwarzanie schematów i metryki jakości.  
	[Dokumentacja »](reference/schematics/metrics.md)

</div>

> Wskazówka: wyszukiwarka w górnym pasku szybko znajdzie klasy/funkcje. Przykładowe obrazy znajdziesz w `charts/charts_examples/`, a dane CSV w `charts/dane/`.

