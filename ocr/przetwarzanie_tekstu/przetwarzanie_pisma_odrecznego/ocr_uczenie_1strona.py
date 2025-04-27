import os
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from torch.utils.data import Dataset

# -------------------------
# USTAWIENIA
# -------------------------
BASE_MODEL    = "microsoft/trocr-base-handwritten"
OUTPUT_DIR    = "trocr_1strona"
PAGE_IMAGE    = "SMHD-forms\Notes Based-0500-0530\0512.jpg"       # zmień na ścieżkę do swojego pliku
PAGE_TEXT     = "SMHD-forms\Notes Based-0500-0530\0512.txt"       # ścieżka do odpowiadającego txt
NUM_EPOCHS    = 50               # ile razy „przerobić” stronę
SAVE_STEPS    = 100              # checkpoint co 100 batchy
SAVE_LIMIT    = 1                # trzymaj tylko ostatni checkpoint
BATCH_SIZE    = 1                # batch=1 na jedną stronę
LEARNING_RATE = 5e-5
MAX_LENGTH    = 512              # maks. długość tekstu po tokenizacji

# -------------------------
# Dataset 1-stronny
# -------------------------
class OnePageDataset(Dataset):
    def __init__(self, image_path, text_path, processor):
        self.processor = processor
        self.image = Image.open(image_path).convert("RGB")
        self.text  = Path(text_path).read_text(encoding="utf-8", errors="ignore").replace("\n", " ")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        enc = self.processor(
            images=self.image,
            text=self.text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        return {
            "pixel_values": enc.pixel_values.squeeze(0),
            "labels":       enc.labels.squeeze(0),
        }

# -------------------------
# Load model + processor
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading base model {BASE_MODEL} on {device}")
processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
model     = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL).to(device)

# konieczne ustawienia tokenów
tok = processor.tokenizer
model.config.decoder_start_token_id = tok.bos_token_id
model.config.eos_token_id           = tok.eos_token_id
model.config.pad_token_id           = tok.pad_token_id
model.config.vocab_size             = len(tok)

model.train()

# -------------------------
# Przygotuj dataset
# -------------------------
print("Preparing one-page dataset")
dataset = OnePageDataset(PAGE_IMAGE, PAGE_TEXT, processor)

# -------------------------
# TrainingArguments + Trainer
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_LIMIT,
    logging_steps=SAVE_STEPS,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,
)

# -------------------------
# Fine-tuning
# -------------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Finished fine-tuning — model saved to", OUTPUT_DIR)
