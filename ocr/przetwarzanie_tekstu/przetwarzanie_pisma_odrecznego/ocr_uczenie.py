import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Trainer,
    TrainingArguments
)

# -------------------------
# 1) Ustawienia
# -------------------------
DATA_DIR   = "SMHD-forms"                              # katalog z datasetem
MODEL_NAME = "microsoft/trocr-large-handwritten"       # bazowy model
OUTPUT_DIR = "trocr_smhdf_nauczony"                   # gdzie zapisać fine-tuned
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 5
BATCH_SIZE = 4
MAX_LEN    = 256

# -------------------------
# 2) Wczytaj model + processor
# -------------------------
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

# Ustawiamy wymagane atrybuty w config:
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id           = processor.tokenizer.eos_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id

model.to(DEVICE)

# -------------------------
# 3) Segmentacja linii (horizontal projection)
# -------------------------
def segment_lines(img: np.ndarray, min_height: int = 10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hist = np.sum(bw, axis=1)
    thresh = hist.max() * 0.05

    lines, start = [], None
    for y, val in enumerate(hist):
        if val > thresh and start is None:
            start = y
        elif val <= thresh and start is not None:
            if y - start >= min_height:
                lines.append(img[start:y, :])
            start = None
    if start is not None and img.shape[0] - start >= min_height:
        lines.append(img[start:, :])
    return lines

# -------------------------
# 4) Zbierz pary (page.jpg, page.txt)
# -------------------------
pairs = []
for root, _, files in os.walk(DATA_DIR):
    for fn in sorted(files):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, fn)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                pairs.append((img_path, txt_path))

# -------------------------
# 5) Przygotuj (line_img, line_txt)
# -------------------------
examples = []
for img_path, txt_path in pairs:
    img = np.array(Image.open(img_path).convert("RGB"))
    lines_img = segment_lines(img)

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    lines_txt = [l.strip() for l in raw.splitlines() if l.strip()]

    # obetnij do najmniejszej liczby
    n = min(len(lines_img), len(lines_txt))
    for i in range(n):
        examples.append((lines_img[i], lines_txt[i]))

print(f"🔎 Przygotowano {len(examples)} przykładów do treningu.")

# -------------------------
# 6) PyTorch Dataset
# -------------------------
class HTRDataset(torch.utils.data.Dataset):
    def __init__(self, examples, processor):
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img_line, txt_line = self.examples[idx]
        enc = self.processor(
            images=Image.fromarray(img_line),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )
        pixel_values = enc.pixel_values.squeeze(0).to(DEVICE)
        labels = self.processor.tokenizer(
            txt_line,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        ).input_ids
        labels = torch.tensor(labels).to(DEVICE)
        return {"pixel_values": pixel_values, "labels": labels}

dataset = HTRDataset(examples, processor)

# -------------------------
# 7) Data collator
# -------------------------
def collate_fn(batch):
    pv = torch.stack([b["pixel_values"] for b in batch])
    lb = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pv, "labels": lb}

# -------------------------
# 8) TrainingArguments + Trainer
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)

# -------------------------
# 9) Start
# -------------------------
if __name__ == "__main__":
    print("Start fine-tuningu...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Gotowe! Model zapisany w: {OUTPUT_DIR}")
