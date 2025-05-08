import cv2
import numpy as np

def preprocess_for_small_text(image, upscale_factor=2.0):
    """
    Applies simple preprocessing: upscaling and slight Gaussian blur.
    Returns an upscaled grayscale image.
    """
    if upscale_factor > 1.0:
        height, width = image.shape[:2]
        new_dim = (int(width * upscale_factor), int(height * upscale_factor))
        processed_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)
    else:
        processed_image = image

    if len(processed_image.shape) == 3:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_image

    kernel = np.ones((2, 2), np.uint8)
    dilated_gray = cv2.dilate(gray, kernel, iterations=1)

    blurred_gray = cv2.GaussianBlur(dilated_gray, (3, 3), 0)

    return blurred_gray

if __name__ == "__main__":
    # Test funkcji na przykładowym obrazie
    import sys
    import os
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Błąd: Plik {image_path} nie istnieje")
            sys.exit(1)
            
        output_path = image_path.rsplit(".", 1)[0] + "_preprocessed.png"
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Błąd: Nie udało się odczytać obrazu {image_path}")
            sys.exit(1)
            
        processed = preprocess_for_small_text(image)
        cv2.imwrite(output_path, processed)
        print(f"Zapisano przetworzony obraz do: {output_path}")
    else:
        print("Użycie: python chart_preprocessing.py <ścieżka_do_obrazu>")
        sys.exit(1)

