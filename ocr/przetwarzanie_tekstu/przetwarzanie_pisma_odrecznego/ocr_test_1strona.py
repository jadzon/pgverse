import sys
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# 1) Ścieżki i urządzenia
MODEL_DIR = "trocr_onepage_ft"                  # folder z wytrenowanym modelem
BASE_PROC = "microsoft/trocr-base-handwritten"  # oryginalny processor
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN   = 256
NUM_BEAMS = 1

# 2) Załaduj model i procesor
print("Ładuję model i processor…")
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
processor = TrOCRProcessor.from_pretrained(BASE_PROC)

# Upewnij się, że tokeny są poprawnie ustawione w configu:
tok = processor.tokenizer
model.config.decoder_start_token_id = tok.bos_token_id
model.config.eos_token_id           = tok.eos_token_id
model.config.pad_token_id           = tok.pad_token_id
model.eval()

# 3) Funkcja OCR
@torch.no_grad()
def ocr(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(
        images=img,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    ).pixel_values.to(DEVICE)
    out_ids = model.generate(inputs, max_length=MAX_LEN, num_beams=NUM_BEAMS)
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]

# 4) CLI
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie: python test_onepage.py <ścieżka_do_obrazka.png>")
        sys.exit(1)
    path = sys.argv[1]
    print(f"Rozpoznaję: {path}")
    res = ocr(path)
    print("\nWynik OCR:\n")
    print(res)
