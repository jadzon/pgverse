import os
import cv2
import numpy as np

def process_image(input_filename="image3.png", grayscale_output="grayscale.png", filtered_output="filtered.png", adjusted_output="adjusted.png", sharpened_output="sharpened.png", alpha=0.8, background_color=250, max_resolution=1024):
    # Sprawdzenie, czy plik istnieje
    if not os.path.isfile(input_filename):
        print(f"Plik {input_filename} nie istnieje.")
        return
    
    # Wczytanie obrazu w skali szarości
    image = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Błąd wczytywania obrazu.")
        return
    
    # Resize the image to have the longer side at most max_resolution pixels
    height, width = image.shape
    if max(height, width) > max_resolution:
        if height > width:
            new_height = max_resolution
            new_width = int((width / height) * max_resolution)
        else:
            new_width = max_resolution
            new_height = int((height / width) * max_resolution)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Zapisanie obrazu w skali szarości
    cv2.imwrite(grayscale_output, image)
    print(f"Obraz w skali szarości zapisano jako {grayscale_output}")
    
    # Nałożenie filtra Gaussa dla redukcji szumu (parametry dostosowane do poziomu <2%)
    filtered_image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Zapisanie przefiltrowanego obrazu
    cv2.imwrite(filtered_output, filtered_image)
    print(f"Obraz po filtrze Gaussa zapisano jako {filtered_output}")
    
    # Obliczenie średniej wartości pikseli
    mean_pixel_value = np.mean(filtered_image)
    
    # Korekcja jasności i kontrastu
    beta = 127.5 - (mean_pixel_value * alpha)
    adjusted_image = cv2.convertScaleAbs(filtered_image, alpha=alpha, beta=beta)
    
    # Zapisanie obrazu po korekcji jasności i kontrastu
    cv2.imwrite(adjusted_output, adjusted_image)
    print(f"Obraz po korekcji jasności i kontrastu zapisano jako {adjusted_output}")
    
    # Sharpening the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(adjusted_image, -1, kernel)
    
    # Zmiana koloru tła na 250-255
    _, mask_high = cv2.threshold(sharpened_image, 100, 255, cv2.THRESH_BINARY)
    sharpened_image[mask_high == 255] = background_color
    
    # Zmiana koloru tła na 0-50
    _, mask_low = cv2.threshold(sharpened_image, 100, 255, cv2.THRESH_BINARY_INV)
    sharpened_image[mask_low == 255] = np.random.randint(0, 51, size=sharpened_image[mask_low == 255].shape)
    
    # Zapisanie wyostrzonego obrazu
    cv2.imwrite(sharpened_output, sharpened_image)
    print(f"Obraz po wyostrzeniu zapisano jako {sharpened_output}")

if __name__ == "__main__":
    process_image()