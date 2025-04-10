from transformers import pipeline, AutoTokenizer
import os
import torch
import re
import argparse

def extract_summary(text, prompt_instruction, use_cuda=False, max_input_length=500):
    """
    Funkcja tworząca streszczenie tekstu przy użyciu modelu BART.
    
    Args:
        text: Tekst do streszczenia
        prompt_instruction: Instrukcja dla modelu, co ma streszczać
        use_cuda: Czy używać GPU (True) czy CPU (False)
        max_input_length: Maksymalna długość wejścia w tokenach
    
    Returns:
        Streszczenie tekstu
    """
    # Zawsze używaj CPU, ponieważ mamy problem z CUDA
    device = -1
    print(f"Device set to use {'cuda' if device >= 0 else 'cpu'}")
    
    try:
        # Inicjalizacja pipeline do streszczenia i tokenizera
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer, device=device)
        
        # Przygotuj pełny tekst z promptem
        input_text = f"{prompt_instruction}\n\n{text}"
        
        # Tokenizuj tekst, aby sprawdzić jego rzeczywistą długość
        tokens = tokenizer(input_text, return_tensors="pt", truncation=False, add_special_tokens=True)
        token_count = len(tokens['input_ids'][0])
        
        # Jeśli tekst jest zbyt długi, podziel go na części
        if token_count > max_input_length:
            print(f"Tekst ma {token_count} tokenów, co przekracza limit {max_input_length}. Dzielenie na fragmenty...")
            
            # Dzielimy tekst na akapity
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Grupujemy akapity tak, aby każda grupa miała ok. max_input_length tokenów
            chunks = []
            current_chunk = []
            current_tokens = 0
            prompt_tokens = len(tokenizer(prompt_instruction, return_tensors="pt")['input_ids'][0])
            
            for para in paragraphs:
                para_tokens = len(tokenizer(para, return_tensors="pt")['input_ids'][0])
                
                # Sprawdź czy dodanie akapitu nie przekroczy limitu
                if current_tokens + para_tokens + prompt_tokens > max_input_length and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
                    
            # Dodaj ostatni chunk jeśli istnieje
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
            print(f"Podzielono tekst na {len(chunks)} fragmentów")
            
            # Przetwarzamy każdy chunk oddzielnie
            summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Przetwarzanie fragmentu {i+1}/{len(chunks)}...")
                full_input = f"{prompt_instruction}\n\n{chunk}"
                
                try:
                    # Zastosuj truncation, aby upewnić się, że nie przekroczymy limitu
                    encoded = tokenizer(full_input, return_tensors="pt", truncation=True, 
                                      max_length=max_input_length)
                    
                    # Generuj streszczenie z mniejszą długością dla mniejszych fragmentów
                    max_length = min(100, int(len(encoded['input_ids'][0]) * 0.4))
                    min_length = min(30, int(len(encoded['input_ids'][0]) * 0.1))
                    
                    summary = summarizer(full_input, max_length=max_length, 
                                       min_length=min_length, do_sample=False,
                                       truncation=True)
                                       
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    print(f"Błąd podczas przetwarzania fragmentu {i+1}: {str(e)}")
                    summaries.append(f"[Nie udało się przetworzyć fragmentu {i+1}]")
                
            return ' '.join(summaries)
        else:
            # Jeśli tekst mieści się w limicie, przetwarzaj całość
            encoded = tokenizer(input_text, return_tensors="pt", truncation=True, 
                              max_length=max_input_length)
            
            max_length = min(150, int(len(encoded['input_ids'][0]) * 0.4))
            min_length = min(40, int(len(encoded['input_ids'][0]) * 0.1))
            
            summary = summarizer(input_text, max_length=max_length, 
                               min_length=min_length, do_sample=False,
                               truncation=True)
            return summary[0]['summary_text']
            
    except Exception as e:
        return f"Wystąpił błąd podczas generowania streszczenia: {str(e)}"

def extract_summary_from_segments(segmented_file_path="output/segmented_output.txt", prompt_instruction="", use_cuda=True, max_input_length=500):
    """
    Funkcja tworząca streszczenie tekstu na podstawie już posegmentowanego pliku.
    
    Args:
        segmented_file_path: Ścieżka do pliku z posegmentowanym tekstem
        prompt_instruction: Instrukcja dla modelu, co ma streszczać
        use_cuda: Czy używać GPU (True) czy CPU (False)
        max_input_length: Maksymalna długość wejścia w tokenach
    
    Returns:
        Streszczenie tekstu
    """
    # Używaj GPU jeśli dostępne i use_cuda=True
    device = 0 if use_cuda and torch.cuda.is_available() else -1
    print(f"Device set to use {'cuda' if device >= 0 else 'cpu'}")
    
    try:
        # Inicjalizacja pipeline do streszczenia i tokenizera
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer, device=device)
        
        # Wczytaj plik z posegmentowanym tekstem
        with open(segmented_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Podziel na segmenty po pustych liniach (akapitach)
        segments = re.split(r'\n\s*\n', content)
        valid_segments = [segment.strip() for segment in segments if segment.strip()]
        
        print(f"Znaleziono {len(valid_segments)} segmentów do przetworzenia")
        
        # Łączymy segmenty w większe fragmenty, aby uzyskać bardziej sensowne streszczenia
        combined_segments = []
        current_segment = ""
        current_tokens = 0
        max_tokens_per_chunk = max_input_length - 100  # Zostawiamy miejsce na prompt
        
        for segment in valid_segments:
            segment_tokens = len(tokenizer.encode(segment))
            
            if current_tokens + segment_tokens > max_tokens_per_chunk and current_segment:
                combined_segments.append(current_segment)
                current_segment = segment
                current_tokens = segment_tokens
            else:
                if current_segment:
                    current_segment += "\n\n" + segment
                else:
                    current_segment = segment
                current_tokens += segment_tokens
        
        # Dodaj ostatni segment, jeśli istnieje
        if current_segment:
            combined_segments.append(current_segment)
        
        print(f"Połączono segmenty w {len(combined_segments)} większych fragmentów")
        
        # Przetwarzamy każdy połączony segment
        summaries = []
        for i, segment in enumerate(combined_segments):
            print(f"Przetwarzanie fragmentu {i+1}/{len(combined_segments)}...")
            
            # Uproszczony prompt, który nie będzie powodował problemów
            input_text = f"{segment}"
            
            try:
                # Zastosuj truncation, aby upewnić się, że nie przekroczymy limitu
                summary = summarizer(
                    input_text, 
                    max_length=150,
                    min_length=50, 
                    do_sample=False,
                    truncation=True
                )
                
                # Jeśli podano instrukcję, dodajemy drugi krok przetwarzania
                if prompt_instruction:
                    refined_prompt = f"Zgodnie z instrukcją: {prompt_instruction}\n\n{summary[0]['summary_text']}"
                    refined_summary = summarizer(
                        refined_prompt,
                        max_length=150,
                        min_length=40,
                        do_sample=False,
                        truncation=True
                    )
                    summaries.append(refined_summary[0]['summary_text'])
                else:
                    summaries.append(summary[0]['summary_text'])
                    
            except Exception as e:
                print(f"Błąd podczas przetwarzania fragmentu {i+1}: {str(e)}")
                # Dodaj krótki fragment samego segmentu w przypadku błędu
                segment_preview = segment[:100] + "..." if len(segment) > 100 else segment
                summaries.append(f"[Błąd przetwarzania fragmentu {i+1}]")
        
        # Połącz wszystkie streszczenia w jeden tekst
        combined_summary = "\n\n".join(summaries)
        return combined_summary
            
    except Exception as e:
        return f"Wystąpił błąd podczas generowania streszczenia: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generowanie streszczeń tekstu z segmentowanego pliku")
    parser.add_argument("--input", default="output/segmented_output.txt", help="Plik z posegmentowanym tekstem")
    parser.add_argument("--prompt", default="Przedstaw kluczowe informacje z tekstu.",
                       help="Instrukcja (prompt) dla streszczenia")
    parser.add_argument("--max_length", type=int, default=500, help="Maksymalna długość fragmentu tekstu")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Użyj GPU do przetwarzania")
    args = parser.parse_args()
    
    # Ustal ścieżkę do pliku wejściowego - najpierw sprawdź czy plik istnieje
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"Plik {args.input} nie istnieje.")
        print("Upewnij się, że plik znajduje się we wskazanej lokalizacji.")
        exit(1)
    
    # Utwórz folder output, jeśli nie istnieje
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_ex.txt")
    
    print(f"Rozpoczynam generowanie streszczenia z pliku {input_file}...")
    print(f"Instrukcja dla streszczenia: '{args.prompt}'")
    
    # Włącz GPU domyślnie
    summary = extract_summary_from_segments(
        input_file, 
        args.prompt, 
        use_cuda=args.use_gpu, 
        max_input_length=args.max_length
    )
    
    # Zapisz wyniki do pliku
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"📌 Streszczenie tekstu na podstawie instrukcji: '{args.prompt}'\n\n")
        f.write(summary)
    
    print("\n📌 Streszczenie zostało utworzone i zapisane do pliku:")
    print(f"   {output_path}")
