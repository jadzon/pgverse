from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
# Wyczyść pamięć CUDA
torch.cuda.empty_cache()

# 1. Załaduj model ViT-GPT2 (znacznie lżejszy niż BLIP2)
model_name = "nlpconnect/vit-gpt2-image-captioning"
image_processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Przenieś model na GPU w trybie oszczędnym
model = model.to("cuda").half()  # half precision dla oszczędności pamięci

# 2. Wczytaj obraz
image = Image.open("input/image2.webp").convert("RGB")

# 3. Przygotuj wejście do modelu
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to("cuda")

# 4. Generuj opis
with torch.no_grad():  # Wyłącz śledzenie gradientów
    output_ids = model.generate(
        pixel_values,
        max_length=30,
        num_beams=4,
        early_stopping=True
    )
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Szczegółowy opis (EN):", caption)

# 5. Tłumaczenie na polski


# Załaduj model tłumaczeniowy
m2m_model_name = "facebook/m2m100_1.2B"
m2m_tokenizer = M2M100Tokenizer.from_pretrained(m2m_model_name)
m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name).to("cuda").half()

# Ustaw język źródłowy i docelowy
src_lang = "en"
tgt_lang = "pl"
m2m_tokenizer.src_lang = src_lang

# Tłumacz tekst
translation_inputs = m2m_tokenizer(caption, return_tensors="pt").to("cuda")
with torch.no_grad():
    translated_ids = m2m_model.generate(
        **translation_inputs,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(tgt_lang),
        max_length=50,
        num_beams=4,
        early_stopping=True
    )
translated_caption = m2m_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

print("Szczegółowy opis (PL):", translated_caption)