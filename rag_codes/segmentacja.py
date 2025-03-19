from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Nazwa modelu (upewnij się, że masz go pobranego lub dostępnego przez Hugging Face)
MODEL_NAME = "meta-llama/Llama-2-27b-hf"

# Wczytanie modelu i tokenizera
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Wczytanie tekstu z pliku
with open("input.txt", "r", encoding="utf-8") as f:
    user_text = f.read().strip()

# Wstawienie tekstu w szablon prompta
prompt_template = f"""Przeczytaj poniższy tekst i podziel go na logiczne sekcje. 
Dla każdej sekcji nadaj tytuł oraz krótki opis najważniejszych punktów.

Tekst:
{user_text}
"""

# Tokenizacja i generowanie odpowiedzi
inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)

# Dekodowanie wyniku
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Zapis do pliku
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(response)

print(" Wygenerowana odpowiedź zapisana do output.txt")