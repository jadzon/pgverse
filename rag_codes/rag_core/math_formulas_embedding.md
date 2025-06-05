# Uzyskiwanie embeddingów dla wzorów matematycznych
### Kompleksowe porównanie podejść i uzasadnienie wyboru GNN

<h3 align="center">Obecny Stan Wiedzy - Trzy Kluczowe Podejścia </h3>   
<h4>1. Symbol2Vec/Formula2Vec (2017) -  Badacze z Uniwersytetu Pekińskiego jako pierwsi zastosowali techniki neural embedding do wzorów matematycznych. Ich podejście bazowało na CBOW (Continuous Bag of Words) z negative sampling dla 892 symboli LaTeX oraz Paragraph Vectors (PV-DM) dla całych wzorów.</h4>

## Kluczowe sukcesy

    Liczby, zmienne i funkcje trygonometryczne grupowały się semantycznie poprawnie

    Udowodniono, że neural approaches działają dla "języka matematycznego"

    Pierwszy empiryczny dowód potencjału embeddingów matematycznych

<h3 align="center">Fundamentalny problem </h3> 
Operatory jak "+" miały "chaotyczne" (messy) embeddingi - najbliższe sąsiedztwo to losowe symbole. Autorzy zdiagnozowali przyczynę: "wzory są bardzo strukturalne, semantyczne znaczenie symbolu operatora nie jest podobne do jego liniowych symboli sąsiadujących (zwykle są to liczby lub zmienne)".

<h4>2.BERT- Based Formula Embedding (2021) 
Zespół z NIT Silchar zastosował BERT-Base do embeddingu wzorów LaTeX, traktując je jako sekwencje tokenów. Pomimo znaczącej poprawy wyników podejście to odziedziczyło fundamentalne problemy z Symbol2Vec.</h4>

    Zalety: 
    
    Prosta implementacja z wykorzystaniem gotowych modeli

    Wymierne poprawki wydajności

    Kompatybilność z ekosystemem transformerów

    Wady:

    Sekwencyjne przetwarzanie niszczy hierarchię matematyczną

    Preprocessing (konwersja do małych liter) traci semantykę (A² ≠ a²)

    x² + y² i a² + b² otrzymują różne embeddingi mimo identycznej struktury

<h4>3.RAG Math (2023) - Kontekst i Preferencje Człowieka
Digital Harbor Foundation skupiła się na generowaniu odpowiedzi z wykorzystaniem zewnętrznych źródeł wiedzy (OpenStax Prealgebra). Badanie ujawniło kluczowe spostrzeżenia dotyczące preferencji edukacyjnych</h4>

    Kluczowe odkrycia:

    Ludzie preferują "low guidance" - umiarkowane wsparcie materiałem źródłowym

    70.6% zapytań ma tematycznie istotne dokumenty, ale tylko 33.3% rzeczywiście użyteczne

    Trade-off między groundedness a preferencjami użytkowników

    Ograniczenia:

    Brak strukturalnego rozumienia matematyki

    Zależność od jakości korpusu źródłowego

    Istotność informacji nie przewiduje preferencji człowieka

## Co to jest GNN?

**Graph Neural Network** to sieć neuronowa do pracy z grafami. Agreguje informacje z sąsiadujących węzłów i uczy się relacji między elementami.

## Dlaczego GNN z AST?

### 1. Zachowanie Struktury
Wzory matematyczne mają naturalną hierarchię - GNN może ją wykorzystać lepiej niż modele sekwencyjne.

### 2. Semantyka Matematyczna
GNN rozpoznaje, że:
- `a² + b²` (suma kwadratów) ≠ `(a + b)²` (kwadrat sumy)
- Struktura AST pokazuje różnicę w priorytecie operacji



<h3>Przewagi nad wszystkimi poprzednimi podejściami</h3>

    Vs Symbol2Vec:
    Rozwiązuje problem "chaotycznych operatorów" przez kontekst strukturalny

    Zachowuje sukcesy (dobre embeddingi liczb, funkcji)

    Implementuje postulowaną tree-based tokenizację

    Vs BERT:
    Eliminuje problemy sekwencyjnego przetwarzania

    Zachowuje wydajność transformerów przez Graph Attention

    Semantyczna równoważność wzorów strukturalnie identycznych

    Vs RAG:
    Dodaje strukturalne rozumienie matematyki

    Zachowuje spostrzeżenia o ludzkich preferencjach

<h3>References</h3>

Dadure, P., Pakray, P., & Bandyopadhyay, S. (2021). BERT-based formula embedding model to facilitated formula retrieval in ARQMath2 tasks.


Levonian, Z., Li, C., Zhu, W., Gade, A., Henkel, O., Postle, M. E., & Xing, W. (2023). Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference.

Gao, L., Yuan, K., Jiang, Z., Yan, Z., Yin, Y., & Tang, Z. (2017). Preliminary Exploration of Formula Embedding for Mathematical Information Retrieval: Can Mathematical Formulae Be Embedded like A Natural Language?


