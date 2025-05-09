from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

# Wczytaj model GIT
model_name = "microsoft/git-large"  # lub "microsoft/git-base"
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Załaduj obraz
image = Image.open("input/image2.webp")  # lub z BytesIO jeśli to PDF extract

# Tokenizacja + przygotowanie wejścia
# Używamy legacy=True, ponieważ dostarczamy tylko obraz (bez tekstu)
inputs = processor(images=image, return_tensors="pt", legacy=True)

# Generacja bardziej szczegółowego opisu
generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    max_length=150,  # Zwiększona maksymalna długość
    num_beams=5,     # Beam search dla lepszych wyników
    do_sample=True,  # Włączenie próbkowania dla temperature i top_p
    temperature=0.7, # Dodanie kreatywności
    top_p=0.9,       # Próbkowanie nucleus
    length_penalty=1.0,  # Zachęcanie do generowania dłuższych opisów
    no_repeat_ngram_size=3  # Unikanie powtórzeń
)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Szczegółowy opis:", generated_caption)