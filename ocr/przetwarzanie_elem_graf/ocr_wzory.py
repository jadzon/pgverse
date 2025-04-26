from pix2text import Pix2Text
from PIL import Image
import json
import numpy as np

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj

def get_bounding_box(item):


    # "position", wyznacz prostokątny bounding box
   if 'position' in item:
        positions = item['position']  # lista punktów [ [x, y], [x, y], ... ]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return [x1, y1, x2, y2]


# === Ścieżki ===
image_path = 'wzor.png'
analysis_json = 'manual_detected.json'

# === Inicjalizacja Pix2Text
p2t = Pix2Text.from_config()

# === Rozpoznanie wzorów i zapis do JSON (konwersja NumPy)
detected_items = p2t.recognize(
    image_path,
    file_type='text_formula',
    return_text=False
)
serializable_items = make_json_serializable(detected_items)
with open(analysis_json, 'w', encoding='utf-8') as f:
    json.dump(serializable_items, f, ensure_ascii=False, indent=2)

# === Wycinanie elementów typu 'isolated' korzystając z 'position' lub 'box'
print("\n✂️ Wycinanie wzorów typu 'isolated':")
with open(analysis_json, 'r', encoding='utf-8') as f:
    loaded_items = json.load(f)

with Image.open(image_path) as img:
    for i, item in enumerate(loaded_items):
        if item.get('type') == 'isolated':
            bbox = get_bounding_box(item)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = img.crop((x1, y1, x2, y2)).copy()
                crop.save(f'formula_isolated_{i}.png')
                text_preview = item.get('text', '')[:60]
                print(f"✅ Wycięto formula_isolated_{i}.png — {text_preview}...")
            else:
                print(f"⏭ Pominięto element {i} — brak 'position' lub 'box'")
        else:
            print(f"⏭ Pominięto element {i} — typ nie jest 'isolated'")
