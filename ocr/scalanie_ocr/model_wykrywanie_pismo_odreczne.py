import argparse
import logging
from pathlib import Path

import torch
from pdf2image import convert_from_path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

def setup_logger():
    """
    Funkcjonalność:
        Konfiguruje logger do wyświetlania komunikatów programu.

    Args:
        Brak

    Returns:
        None
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO
    )

def parse_args():
    """
    Funkcjonalność:
        Parsuje argumenty wiersza poleceń i zwraca je jako obiekt Namespace.

    Args:
        Brak (argumenty pobierane z sys.argv)

    Returns:
        argparse.Namespace - obiekt z parametrami:
            pdf_folder : Path - katalog z plikami PDF
            txt_folder : Path - katalog wyjściowy na pliki TXT
            dpi : int - rozdzielczość konwersji PDF→obraz
            model : str - nazwa modelu HuggingFace
            max_tokens : int - limit tokenów dla generacji
    """
    p = argparse.ArgumentParser(
        description="Szybki OCR ręcznego pisma PDF → TXT (czyste Florence)"
    )
    p.add_argument(
        "-i", "--pdf-folder",
        type=Path,
        default=Path("pdfs"),
        help="Folder z PDF-ami (domyślnie ./pdfs)"
    )
    p.add_argument(
        "-o", "--txt-folder",
        type=Path,
        default=Path("texts"),
        help="Folder na TXT-y (domyślnie ./texts)"
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI konwersji PDF→obraz (domyślnie 200, mniej = szybsze)"
    )
    p.add_argument(
        "--model",
        type=str,
        default="microsoft/Florence-2-large",
        help="Model HF (możesz podmienić na microsoft/Florence-2-base)"
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="max_new_tokens (domyślnie 512)"
    )
    return p.parse_args()

def extract_ocr(
    pdf_folder: Path,
    txt_folder: Path,
    dpi: int,
    model_name: str,
    max_tokens: int
):
    """
    Funkcjonalność:
        Wykonuje OCR dla wszystkich plików PDF w katalogu i zapisuje teksty do plików TXT.

    Args:
        pdf_folder : Path - katalog z plikami PDF
        txt_folder : Path - katalog docelowy na pliki TXT
        dpi : int - rozdzielczość konwersji PDF→obraz
        model_name : str - nazwa modelu na HuggingFace
        max_tokens : int - maksymalna liczba tokenów generowanych przez model

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    logging.info(f"Urządzenie={device}, dtype={dtype}, model={model_name}")

    # Ładujemy procesor i model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device).eval()

    # Przygotowujemy folder na wyniki
    txt_folder.mkdir(parents=True, exist_ok=True)

    # Przetwarzamy każdy PDF
    for pdf_path in sorted(pdf_folder.glob("*.pdf")):
        logging.info(f"=== {pdf_path.name} ===")
        try:
            pages = convert_from_path(str(pdf_path), dpi=dpi)
        except Exception as e:
            logging.error(f"Błąd konwersji PDF: {e}")
            continue

        texts = []
        # OCR każdej strony
        for page_idx, img in enumerate(tqdm(pages, desc="Strony", leave=False), start=1):
            # UWAGA: Florence wymaga dokładnie tego tokenu
            inp = processor(text="<OCR>", images=img, return_tensors="pt")
            pixel_values = inp["pixel_values"].to(device).to(dtype)
            input_ids    = inp["input_ids"].to(device)

            # Generacja z Florence
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=max_tokens,
                    num_beams=1,
                    do_sample=False
                )

            # Dekodowanie i postprocessing
            raw = processor.batch_decode(gen, skip_special_tokens=False)[0]
            ocr = processor.post_process_generation(
                raw,
                task="<OCR>",
                image_size=(img.width, img.height)
            ).get("<OCR>", "").strip()

            texts.append(ocr)

        # Zapis do pliku TXT
        out_file = txt_folder / f"{pdf_path.stem}.txt"
        out_file.write_text(
            "\n\n".join(f"--- Strona {i} ---\n{t}"
                        for i, t in enumerate(texts, start=1)),
            encoding="utf-8"
        )
        logging.info(f"Zapisano → {out_file.name}")

if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    extract_ocr(
        pdf_folder=args.pdf_folder,
        txt_folder=args.txt_folder,
        dpi=args.dpi,
        model_name=args.model,
        max_tokens=args.max_tokens
    )
