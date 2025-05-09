# Programy chunkujące tekst

embedding_chunker.py - stosuje wielojęzykowy model do embeddingów, porównuje za pomocą podobieństwa cosinusowego, co pozwala na dokładne dzielenie tekstu

semantic_chunker.py - wykorzystuje algorytmy klastrowania do wykrywania semantycznej zmiany tematów (wektoryzacja TF-IDF pomaga w analizie semantycznej) do segmentacji tekstu

LLM_chunker.py - wykorzystuje template do prompta i LLM do dokładnej segmentacji (wymaga dużej mocy obliczeniowej)