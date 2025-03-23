import os
import cv2
import numpy as np

def calculate_noise_percentage(original_image, filtered_image, threshold=20):
    """
    Calculates the percentage of noisy pixels in the image.

    Args:
        original_image (numpy.ndarray): The original grayscale image.
        filtered_image (numpy.ndarray): The image after applying the bilateral filter.
        threshold (int): The difference threshold to consider a pixel as noisy.

    Returns:
        float: The percentage of noisy pixels in the image.
    """
    # Calculate the absolute difference between the original and filtered image
    noise_map = cv2.absdiff(original_image, filtered_image)
    
    # Count pixels where the difference exceeds the threshold
    noisy_pixels = np.sum(noise_map > threshold)
    
    # Calculate the total number of pixels
    total_pixels = original_image.size
    
    # Calculate the noise percentage
    noise_percentage = (noisy_pixels / total_pixels) * 100
    
    return noise_percentage

# Example usage in your process_image function
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
    
    #W razie usuwania za mało kratek zwiekszyc d do 8 lub 9 pod spodem
    # Apply Bilateral Filter
    filtered_image = cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)
    
    # Calculate noise percentage
    noise_percentage = calculate_noise_percentage(image, filtered_image)
    print(f"Noise percentage: {noise_percentage:.2f}%")
    
    if noise_percentage < 2:
        print("Noise is below 2%.")
    else:
        print("Noise is above 2%.")
    
    # Continue with the rest of the processing...
    # Obliczenie średniej wartości pikseli
    mean_pixel_value = np.mean(filtered_image)
    
    # Korekcja jasności i kontrastu
    beta = 127.5 - (mean_pixel_value * alpha)
    adjusted_image = cv2.convertScaleAbs(filtered_image, alpha=alpha, beta=beta)
    
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