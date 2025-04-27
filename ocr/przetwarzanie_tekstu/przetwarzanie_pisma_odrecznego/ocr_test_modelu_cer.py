import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# 1) Ścieżki i urządzenia
MODEL_DIR    = "trocr_1strona"               # folder z wytrenowanym modelem
BASE_PROC    = "microsoft/trocr-large-handwritten" # oryginalny processor
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN      = 256

# 2) Załaduj model i procesor
print(f"Loading model from {MODEL_DIR} and processor {BASE_PROC}…")
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
processor = TrOCRProcessor.from_pretrained(BASE_PROC)
tok = processor.tokenizer
model.config.decoder_start_token_id = tok.bos_token_id
model.config.eos_token_id           = tok.eos_token_id
model.config.pad_token_id           = tok.pad_token_id
model.eval()

# 3) Segmentacja
def segment(img: np.ndarray, min_h=10, tol=0.05):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    hist = bw.sum(axis=1)
    thr  = hist.max() * tol
    lines, st = [], None
    for y, v in enumerate(hist):
        if v > thr and st is None:
            st = y
        elif v <= thr and st is not None:
            if y - st >= min_h:
                lines.append(img[st:y, :])
            st = None
    if st is not None and img.shape[0] - st >= min_h:
        lines.append(img[st:, :])
    return lines

# 4) CER
def cer(ref: str, hyp: str) -> float:
    r, h = ref, hyp
    R, H = len(r), len(h)
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (r[i-1] != h[j-1])
            )
    return dp[R][H] / max(R, 1)

# Main
def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_cer.py <path_to_image.jpg>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    txt_path = os.path.splitext(img_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        print(f"Ground-truth .txt not found for {img_path}")
        sys.exit(1)

    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        gt_lines = [l.strip() for l in f if l.strip()]
    print(f"Found {len(gt_lines)} ground-truth lines in {os.path.basename(txt_path)}")

    np_img = np.array(Image.open(img_path).convert("RGB"))
    segs   = segment(np_img)
    print(f"Detected {len(segs)} text lines via segmentation\n")

    n = min(len(gt_lines), len(segs))
    if n == 0:
        print("No lines to evaluate!")
        sys.exit(1)

    total_cer = 0.0
    for i in range(n):
        pil = Image.fromarray(segs[i]).convert("RGB")
        encoding = processor(
            images=pil,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        ).pixel_values.to(DEVICE)
        out_ids = model.generate(
            encoding,
            max_length=MAX_LEN,
            num_beams=1,
            early_stopping=True
        )
        pred = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        score = cer(gt_lines[i], pred)
        total_cer += score

        print(f"[Line {i+1}]")
        print(f" GT: «{gt_lines[i]}»")
        print(f" PR: «{pred}»  —  CER: {score:.2f}\n")

    avg = total_cer / n
    print(f"Average CER over {n} lines: {avg:.2f}")

if __name__ == "__main__":
    main()