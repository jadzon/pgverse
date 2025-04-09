import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama_model():
    """
    Ładuje model LLaMA 7B (lub inny dostępny) z Hugging Face.
    Upewnij się, że masz dostęp do GPU lub zmodyfikuj device_map na 'cpu'.
    """
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Możesz zmienić na inny dostępny model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_tokens=512):
    """
    Generuje odpowiedź modelu LLaMA na podstawie przekazanego prompta.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_info_direct(text, model, tokenizer):
    """
    Podejście Bezpośrednie: LLaMA analizuje tekst i wypisuje kluczowe informacje w jednym kroku.
    """
    prompt = f"""
Przeanalizuj poniższy tekst dotyczący zagadnień inżynierskich i wypisz kluczowe informacje, zwracając szczególną uwagę na:
- Główne tematy i zagadnienia,
- Kluczowe dane liczbowe, parametry techniczne oraz specyfikacje,
- Opisy metod, algorytmów, systemów lub rozwiązań,
- Praktyczne zastosowania oraz przykłady implementacji,
- Wnioski i rekomendacje.

Tekst: \"\"\"{text}\"\"\"
"""
    return generate_response(prompt, model, tokenizer)

def extract_info_questions(text, model, tokenizer):
    """
    Podejście Rozbite na Pytania: LLaMA odpowiada na zestaw konkretnych pytań, co ułatwia uporządkowanie ekstrakcji.
    """
    prompt = f"""
Dla poniższego tekstu dotyczącego zagadnień inżynierskich odpowiedz na następujące pytania:
1. Jakie są główne tematy i zagadnienia poruszane w tekście?
2. Jakie kluczowe dane liczbowe, parametry techniczne lub specyfikacje zostały przedstawione?
3. Jakie metody, algorytmy lub systemy są opisane i jakie mają zastosowanie?
4. Jakie praktyczne przykłady implementacji lub zastosowań zostały wymienione?
5. Jakie wnioski lub rekomendacje można wyciągnąć z tekstu?

Tekst: \"\"\"{text}\"\"\"
"""
    return generate_response(prompt, model, tokenizer)

def compare_extraction_methods(text):
    """
    Funkcja porównuje obie metody ekstrakcji kluczowych informacji:
    1. Podejście Bezpośrednie
    2. Podejście Rozbite na Pytania
    """
    model, tokenizer = load_llama_model()
    
    print("Uruchamiam metodę bezpośrednią...\n")
    direct_result = extract_info_direct(text, model, tokenizer)
    print("--- Podejście Bezpośrednie ---")
    print(direct_result)
    print("\n" + "="*60 + "\n")
    
    print("Uruchamiam podejście rozbite na pytania...\n")
    questions_result = extract_info_questions(text, model, tokenizer)
    print("--- Podejście Rozbite na Pytania ---")
    print(questions_result)
    
    print("\n--- Porównanie Metod ---")
    print("Sprawdź, która metoda lepiej odpowiada Twoim oczekiwaniom pod względem przejrzystości i kompletności informacji.")

if __name__ == '__main__':
    # Przykładowy tekst dotyczący zagadnień inżynierskich
    sample_text = """
W dziedzinie automatyki przemysłowej systemy sterowania oparte na mikrokontrolerach oraz algorytmach PID są powszechnie stosowane.
Nowoczesne urządzenia elektroniczne często wykorzystują procesory ARM, które umożliwiają wysoką wydajność przy niskim poborze mocy.
Przykładem może być zastosowanie czujników temperatury i ciśnienia w systemach HVAC, gdzie dane z czujników są przetwarzane w czasie rzeczywistym, a wyniki służą do regulacji systemu.
W robotyce zaawansowane algorytmy sterowania umożliwiają precyzyjne ruchy manipulatorów, co jest kluczowe w zastosowaniach takich jak montaż czy spawanie.
Integracja systemów IoT z tradycyjnymi systemami sterowania otwiera nowe możliwości optymalizacji procesów przemysłowych.
"""
    compare_extraction_methods(sample_text)
