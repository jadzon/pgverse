import json
from bert_score import score
import enviromental_variables as ev

def compute_and_save_bertscore(query, answer, references, output_path=ev.JSON_PATH):
    """
    Oblicza BERTScore pomiędzy odpowiedzią a każdą referencją.
    Zapisuje wyniki, pytanie, odpowiedź oraz referencje do pliku JSON.
    """
    results = []
    # Dla każdej referencji obliczamy oddzielnie BERTScore
    for ref in references:
        P, R, F1 = score([answer], [ref], lang='pl', verbose=False)
        # P, R, F1 są tensorem, pobieramy wartość
        results.append({
            'reference': ref,
            'precision': float(P[0].item()),
            'recall': float(R[0].item()),
            'f1': float(F1[0].item())
        })

    # Struktura wpisu do pliku
    entry = {
        'query': query,
        'answer': answer,
        'scores': results
    }

    # Wczytanie istniejących wyników (jeżeli plik nie istnieje, tworzymy nową listę)
    try:
        with open(output_path, 'r', encoding='utf-8') as rf:
            data = json.load(rf)
    except FileNotFoundError:
        data = []

    # Dodanie nowego wpisu i zapis
    data.append(entry)
    with open(output_path, 'w', encoding='utf-8') as wf:
        json.dump(data, wf, ensure_ascii=False, indent=2)

    print(f"Zapisano wyniki BERTScore do {output_path}")
