import fitz 
import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import color, exposure, filters, restoration, img_as_float, img_as_ubyte
from imageio import imwrite

pdf_file = "ep.pdf"
output_folder = "output_scikit"
os.makedirs(output_folder, exist_ok=True)
show_comparisons = True

# === Preprocessing ===
def preprocess_scikit(img_rgb):
    """
    Preprocessing obrazu w scikit-image wg best practices:
    1. Konwersja do skali szarości
    2. Denoise (bilateral)
    3. CLAHE (equalizacja histogramu adaptacyjna)
    4. Sharpening (Sobel)
    5. Normalizacja
    """
    img_float = img_as_float(img_rgb)

    # 2. Na grayscale
    img_gray = color.rgb2gray(img_float)

    # 3. Denoising (bilateral filter)
    img_denoised = restoration.denoise_bilateral(img_gray, sigma_color=0.05, sigma_spatial=15)

    # 4. CLAHE (adaptive histogram equalization)
    img_equalized = exposure.equalize_adapthist(img_denoised, clip_limit=0.03)

    # 5. Sharpening (dodajemy krawędzie z Sobela)
    edges = filters.sobel(img_equalized)
    img_sharpened = img_equalized + edges
    img_sharpened = np.clip(img_sharpened, 0, 1)

    # 6. Normalizacja
    img_normalized = (img_sharpened - np.min(img_sharpened)) / (np.max(img_sharpened) - np.min(img_sharpened))

    return img_normalized

def show_comparison(original, processed, title_processed):
    """Porównanie obrazów"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Oryginał")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title_processed)
    plt.axis("off")

    plt.show()

# === Przetwarzanie PDF ===
doc = fitz.open(pdf_file)

for page_num in range(len(doc)):
    print(f"\n Przetwarzanie strony {page_num + 1}")
    
    page = doc.load_page(page_num)
    pix = page.get_pixmap()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Jeśli RGBA, zamień na RGB
    if pix.n >= 4:
        img = img[:, :, :3]

   
    processed_img = preprocess_scikit(img)
   
    processed_img_uint8 = img_as_ubyte(processed_img)
    output_path = os.path.join(output_folder, f"strona_{page_num + 1}_scikit.png")
    imwrite(output_path, processed_img_uint8)

    # === Porównanie obrazów ===
    if show_comparisons:
        show_comparison(img, processed_img, "Przetworzony obraz")

doc.close()

print("\n Obrazy zapisane")