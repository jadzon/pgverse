import re
import numpy as np
from bert_score import score
import torch
import pandas as pd
import matplotlib.pyplot as plt

def load_questions_and_chunks(file_path):
    """Wczytuje pytania, odpowiedzi i chunki z pliku wynikowego."""
    questions_and_chunks = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Dzielimy plik na sekcje dla każdego pytania
    sections = content.split('-' * 80)
    
    for section in sections:
        if not section.strip():
            continue
        
        # Wyodrębniamy pytanie
        question_match = re.search(r'Pytanie: (.*?)\n\nOdpowiedź modelu:', section, re.DOTALL)
        if not question_match:
            continue
        
        question = question_match.group(1).strip()
        
        # Wyodrębniamy odpowiedź modelu - bierzemy tylko właściwą odpowiedź, bez promptu
        full_answer_match = re.search(r'Odpowiedź modelu:(.*?)Znalezione chunki tekstu:', section, re.DOTALL)
        if not full_answer_match:
            continue
        
        full_answer = full_answer_match.group(1).strip()
        
        # Wyodrębniamy właściwą odpowiedź (po promptu)
        # Szukamy pierwszej linii po linii zawierającej {documents=
        answer_lines = full_answer.split('\n')
        actual_answer = ""
        prompt_ended = False
        
        for line in answer_lines:
            if '{documents=' in line:
                prompt_ended = True
                continue
            if prompt_ended and line.strip():
                actual_answer += line + "\n"
        
        if not actual_answer.strip():
            actual_answer = full_answer  # Jeśli nie udało się wyodrębnić, używamy całej odpowiedzi
        
        # Wyodrębniamy chunki tekstu
        chunks = []
        chunk_pattern = r'Chunk (\d+) \(score: ([0-9\.]+)\):\n(.*?)(?=\n\nChunk \d+|\Z)'
        chunk_matches = re.finditer(chunk_pattern, section, re.DOTALL)
        
        for chunk_match in chunk_matches:
            chunk_num = int(chunk_match.group(1))
            score_val = float(chunk_match.group(2))
            text = chunk_match.group(3).strip()
            
            chunks.append({
                'num': chunk_num,
                'score': score_val,
                'text': text
            })
        
        questions_and_chunks.append({
            'question': question,
            'answer': actual_answer.strip(),
            'full_answer': full_answer,
            'chunks': chunks
        })
    
    return questions_and_chunks

def calculate_bertscore_for_chunks(questions_and_chunks):
    """Oblicza metrykę BERTScore dla każdego chunka w odniesieniu do odpowiedzi modelu."""
    results = []
    
    for item in questions_and_chunks:
        question = item['question']
        answer = item['answer']
        chunks = item['chunks']
        
        print(f"Obliczanie BERTScore dla pytania: {question}")
        
        # Przygotowanie list tekstów do oceny
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        if not chunk_texts:
            print("  Brak chunków do oceny.")
            continue
        
        # Obliczenie BERTScore
        try:
            P, R, F1 = score(chunk_texts, [answer] * len(chunk_texts), lang="pl", verbose=True)
            
            # Zapisanie wyników BERTScore dla każdego chunka
            for i, chunk in enumerate(chunks):
                chunk['bertscore'] = {
                    'precision': P[i].item(),
                    'recall': R[i].item(),
                    'f1': F1[i].item()
                }
                
                print(f"  Chunk {chunk['num']} (oryginalny score: {chunk['score']:.4f}):")
                print(f"    Precision: {chunk['bertscore']['precision']:.4f}")
                print(f"    Recall: {chunk['bertscore']['recall']:.4f}")
                print(f"    F1: {chunk['bertscore']['f1']:.4f}")
            
            results.append({
                'question': question,
                'answer': answer,
                'chunks': chunks
            })
            
        except Exception as e:
            print(f"  Błąd podczas obliczania BERTScore: {str(e)}")
    
    return results

def calculate_average_scores(results):
    """Oblicza średnie wartości BERTScore dla wszystkich chunków oraz najlepszych chunków."""
    all_scores = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    best_chunks_scores = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for item in results:
        chunks = item['chunks']
        
        # Dodanie wszystkich wyników do listy
        for chunk in chunks:
            if 'bertscore' in chunk:
                all_scores['precision'].append(chunk['bertscore']['precision'])
                all_scores['recall'].append(chunk['bertscore']['recall'])
                all_scores['f1'].append(chunk['bertscore']['f1'])
        
        # Znalezienie najlepszego wyniku dla pytania (według oryginalnego score)
        if chunks and any('bertscore' in chunk for chunk in chunks):
            best_chunk = max(chunks, key=lambda x: x['score'])
            if 'bertscore' in best_chunk:
                best_chunks_scores['precision'].append(best_chunk['bertscore']['precision'])
                best_chunks_scores['recall'].append(best_chunk['bertscore']['recall'])
                best_chunks_scores['f1'].append(best_chunk['bertscore']['f1'])
    
    # Obliczenie średnich
    avg_all = {
        'precision': np.mean(all_scores['precision']) if all_scores['precision'] else 0,
        'recall': np.mean(all_scores['recall']) if all_scores['recall'] else 0,
        'f1': np.mean(all_scores['f1']) if all_scores['f1'] else 0
    }
    
    avg_best = {
        'precision': np.mean(best_chunks_scores['precision']) if best_chunks_scores['precision'] else 0,
        'recall': np.mean(best_chunks_scores['recall']) if best_chunks_scores['recall'] else 0,
        'f1': np.mean(best_chunks_scores['f1']) if best_chunks_scores['f1'] else 0
    }
    
    return {
        'all_chunks': avg_all,
        'best_chunks': avg_best,
        'all_scores': all_scores,
        'best_chunks_scores': best_chunks_scores
    }

def save_results_to_file(results, averages, output_file):
    """Zapisuje wyniki BERTScore do pliku."""
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("=" * 80 + "\n")
        file.write("WYNIKI ANALIZY BERTSCORE\n")
        file.write("=" * 80 + "\n\n")
        
        for i, item in enumerate(results, 1):
            file.write(f"PYTANIE {i}: {item['question']}\n")
            file.write("-" * 80 + "\n\n")
            
            file.write("ODPOWIEDŹ MODELU:\n")
            file.write(f"{item['answer'][:500]}...\n\n" if len(item['answer']) > 500 else f"{item['answer']}\n\n")
            
            file.write("CHUNKI:\n")
            for chunk in item['chunks']:
                if 'bertscore' in chunk:
                    file.write(f"Chunk {chunk['num']} (oryginalny score: {chunk['score']:.4f}):\n")
                    file.write(f"  Precision: {chunk['bertscore']['precision']:.4f}\n")
                    file.write(f"  Recall: {chunk['bertscore']['recall']:.4f}\n")
                    file.write(f"  F1: {chunk['bertscore']['f1']:.4f}\n")
                    file.write(f"  Tekst: {chunk['text'][:200]}...\n\n" if len(chunk['text']) > 200 else f"  Tekst: {chunk['text']}\n\n")
            
            file.write("\n" + "-" * 80 + "\n\n")
        
        file.write("=" * 80 + "\n")
        file.write("ŚREDNIE WARTOŚCI BERTSCORE\n")
        file.write("=" * 80 + "\n\n")
        
        file.write("Dla wszystkich chunków:\n")
        file.write(f"  Precision: {averages['all_chunks']['precision']:.4f}\n")
        file.write(f"  Recall: {averages['all_chunks']['recall']:.4f}\n")
        file.write(f"  F1: {averages['all_chunks']['f1']:.4f}\n\n")
        
        file.write("Dla najlepszych chunków (według oryginalnego score):\n")
        file.write(f"  Precision: {averages['best_chunks']['precision']:.4f}\n")
        file.write(f"  Recall: {averages['best_chunks']['recall']:.4f}\n")
        file.write(f"  F1: {averages['best_chunks']['f1']:.4f}\n")

def plot_bertscore_vs_original(results, output_file):
    """Tworzy wykres porównujący oryginalny score z BERTScore F1."""
    original_scores = []
    f1_scores = []
    labels = []
    
    for item in results:
        question = item['question']
        for chunk in item['chunks']:
            if 'bertscore' in chunk:
                original_scores.append(chunk['score'])
                f1_scores.append(chunk['bertscore']['f1'])
                labels.append(f"P{results.index(item)+1}C{chunk['num']}")
    
    plt.figure(figsize=(12, 8))
    
    # Wykres punktowy
    plt.scatter(original_scores, f1_scores, alpha=0.7)
    
    # Dodanie etykiet dla punktów
    for i, label in enumerate(labels):
        plt.annotate(label, (original_scores[i], f1_scores[i]), fontsize=8)
    
    # Dodanie linii trendu
    if original_scores and f1_scores:
        z = np.polyfit(original_scores, f1_scores, 1)
        p = np.poly1d(z)
        plt.plot(sorted(original_scores), p(sorted(original_scores)), "r--", alpha=0.7)
    
    plt.xlabel('Oryginalny Score')
    plt.ylabel('BERTScore F1')
    plt.title('Porównanie oryginalnego score z BERTScore F1')
    plt.grid(True, alpha=0.3)
    
    # Zapisanie wykresu
    plt.savefig(output_file)
    plt.close()

def main():
    # Ścieżki do plików
    input_file = 'wyniki_automatyki.txt'
    output_file = 'bertscore_wyniki.txt'
    plot_file = 'bertscore_plot.png'
    
    print(f"Wczytywanie danych z pliku {input_file}...")
    questions_and_chunks = load_questions_and_chunks(input_file)
    print(f"Wczytano {len(questions_and_chunks)} pytań z chunkami.")
    
    print("\nObliczanie metryki BERTScore...")
    results = calculate_bertscore_for_chunks(questions_and_chunks)
    
    print("\nObliczanie średnich wartości...")
    averages = calculate_average_scores(results)
    
    print("\nZapisywanie wyników do pliku...")
    save_results_to_file(results, averages, output_file)
    
    print("\nTworzenie wykresu porównawczego...")
    plot_bertscore_vs_original(results, plot_file)
    
    print("\nŚrednie wartości BERTScore:")
    print("Dla wszystkich chunków:")
    print(f"  Precision: {averages['all_chunks']['precision']:.4f}")
    print(f"  Recall: {averages['all_chunks']['recall']:.4f}")
    print(f"  F1: {averages['all_chunks']['f1']:.4f}")
    
    print("\nDla najlepszych chunków (według oryginalnego score):")
    print(f"  Precision: {averages['best_chunks']['precision']:.4f}")
    print(f"  Recall: {averages['best_chunks']['recall']:.4f}")
    print(f"  F1: {averages['best_chunks']['f1']:.4f}")
    
    print(f"\nWyniki zapisano do pliku {output_file}")
    print(f"Wykres zapisano do pliku {plot_file}")

if __name__ == "__main__":
    main()
