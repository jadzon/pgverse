from rag_codes.rag_functions.latex import FormulaEmbedder

def main():
    # Ścieżka do pliku JSON ze wzorami LaTeX
    json_path = "c:/Users/Maciej/Desktop/ZSDRag/pgverse/rag_codes/subcjects/rezultaty/wzory/latex_wzory.json"
    # Nazwa pliku PNG, pod którą jest zapisany wzór w JSON-ie
    png_filename = "k1_page100_formula1.png"

    # Sposób 1: Używanie statycznej metody (bez potrzeby klucza Cohere)
    print("=== Sposób 1: Statyczna metoda ===")
    embedding = FormulaEmbedder.get_formula_embedding_from_paths(json_path, png_filename)
    if embedding is not None:
        print(f"Embedding: {embedding.shape}")  # oczekiwane (512,)
        print(f"Pierwsze 10 wartości: {embedding[:10]}")
        print(f"Norma wektora: {embedding.dot(embedding)**0.5:.6f}")
    else:
        print("Nie udało się wygenerować embeddingu.")

    print("\n" + "="*60)
    
    # Sposób 2: Test z bezpośrednim wzorem LaTeX
    print("=== Sposób 2: Bezpośredni wzór LaTeX ===")
    test_formula = "\\frac{x^2 + y^2}{z^2}"
    embedding2 = FormulaEmbedder.get_latex_embedding(test_formula)
    if embedding2 is not None:
        print(f"Embedding dla '{test_formula}': {embedding2.shape}")
        print(f"Pierwsze 10 wartości: {embedding2[:10]}")
        print(f"Norma wektora: {embedding2.dot(embedding2)**0.5:.6f}")
    else:
        print("Nie udało się wygenerować embeddingu dla wzoru testowego.")

    print("\n" + "="*60)
    
    # Sposób 3: Używanie instancji klasy (wymaga klucza Cohere - zakomentowane)
    print("=== Sposób 3: Przez instancję klasy (wymaga klucza Cohere) ===")
    print("# WYMAGA KLUCZA COHERE - zakomentowane")
    # cohere_api_key = "twój_klucz_cohere"
    # embedder = FormulaEmbedder(cohere_api_key)
    # embedding3 = embedder.get_formula_by_json_path(json_path, png_filename)
    # if embedding3 is not None:
    #     print(f"Embedding przez instancję: {embedding3.shape}")
    # else:
    #     print("Nie udało się wygenerować embeddingu przez instancję.")

    print("\n" + "="*60)
    
    # Sposób 4: Test z różnymi wzorami (poprawione wzory)
    print("=== Sposób 4: Test z różnymi wzorami ===")
    test_formulas = [
        "x + y + z",  # Usunięto znak równości
        "\\frac{x^2}{y}",  # Prostszy ułamek
        "\\sqrt{x^2 + y^2}",
        "x^2 + 2*x + 1",  # Wielomian
        "\\sin(x) + \\cos(x)"  # Funkcje trygonometryczne
    ]
    
    for i, formula in enumerate(test_formulas, 1):
        print(f"\nTest {i}: {formula}")
        emb = FormulaEmbedder.get_latex_embedding(formula)
        if emb is not None:
            print(f"  ✓ Wymiar: {emb.shape}, Norma: {emb.dot(emb)**0.5:.6f}")
            print(f"  Pierwsze 5 wartości: {emb[:5]}")
        else:
            print(f"  ✗ Błąd generowania embeddingu")

    print("\n" + "="*60)
    
    # Sposób 5: Test z bardzo prostymi wzorami
    print("=== Sposób 5: Test z prostymi wzorami ===")
    simple_formulas = [
        "x",
        "x + y", 
        "x^2",
        "2*x",
        "x/y"
    ]
    
    for i, formula in enumerate(simple_formulas, 1):
        print(f"\nProsty test {i}: {formula}")
        emb = FormulaEmbedder.get_latex_embedding(formula)
        if emb is not None:
            print(f"  ✓ Wymiar: {emb.shape}, Norma: {emb.dot(emb)**0.5:.6f}")
        else:
            print(f"  ✗ Błąd generowania embeddingu")

if __name__ == "__main__":
    print("UWAGA: Aby parser LaTeX działał poprawnie, zainstaluj:")
    print("pip install antlr4-python3-runtime==4.11")
    print("="*60)
    main()