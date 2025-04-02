System RAG dla lektur(plain text)
Aplikacja implementuje system Retrieval-Augmented Generation (RAG) do analizy lektur. Wykorzystuje zaawansowane techniki NLP i uczenia maszynowego do inteligentnego wyszukiwania i generowania odpowiedzi na pytania o tekst.

Główne funkcje programu:
Segmentacja tekstu z adaptacyjnymi progami podobieństwa -> (Wykorzystanie TfidfVectorizer do obliczania podobieństwa między zdaniami, adaptacyjne progi oparte na statystykach tekstu);
Generowanie embedingów z API Cohere -> (Wykorzystanie modelu "embed-multilingual-v3.0" do generowania wektorowych reprezentacji tekstu);
Baza danych grafowa Neo4j do przechowywania i wyszukiwania -> (Przechowywanie fragmentów tekstu jako węzłów z właściwościami tekstowymi i wektorowymi, wykorzystanie indeksów wektorowych do efektywnego wyszukiwania);
Semantyczne wyszukiwanie i reranking wyników -> (Wykorzystanie podobieństwa kosinusowego do wyszukiwania semantycznego, reranking wyników przy użyciu modelu "rerank-multilingual-v3.0" z Cohere);
Generowanie odpowiedzi z wykorzystaniem modelu językowego -> (Wykorzystanie modelu językowego Cohere do generowania odpowiedzi na podstawie kontekstu z wyszukanych fragmentów tekstu);
Analiza wydajności -> (precyzja, recall, F1-score, Obliczanie metryk na podstawie porównania wygenerowanych odpowiedzi z oczekiwanymi wynikami, wykorzystanie klasyfikacji binarnej dla oceny trafności odpowiedz) - WYMAGA dopracowania/zmiany koncepcji ewaluacji;
Wizualizacja metryk wydajności.

