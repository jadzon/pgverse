Aplikacja typu Retrieval-Augmented Generation (RAG), która umożliwia zadawanie pytań dotyczących treści dokumentu tekstowego. Aplikacja wykorzystuje:
    •	Neo4j – baza danych grafowa z indeksem wektorowym do przechowywania embeddingów.
    •	Sentence Transformers – generowanie embeddingów tekstowych.
    •	Hugging Face Transformers (BloomZ) – generowanie odpowiedzi na pytania użytkownika.
Funkcjonalności aplikacji:
    •	Wczytywanie dokumentu tekstowego i dzielenie go na fragmenty (LangChain).
    •	Generowanie embeddingów dla fragmentów tekstu (all-MiniLM-L12-v2).
    •	Przechowywanie embeddingów w bazie danych Neo4j.
    •	Wyszukiwanie podobnych fragmentów tekstowych na podstawie zapytania użytkownika (Similarity search).
    •	Generowanie odpowiedzi za pomocą modelu językowego Hugging Face (BloomZ).

