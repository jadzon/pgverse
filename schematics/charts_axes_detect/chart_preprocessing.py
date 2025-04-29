import cv2
import numpy as np

def preprocess_for_small_text(image, upscale_factor=2.0):
    """
    Applies simple preprocessing: upscaling and slight Gaussian blur.
    Returns an upscaled grayscale image.

    Args:
        image: Input image (NumPy array, expected in BGR format from cv2.imread).
        upscale_factor (float): Factor by which to resize the image initially.

    Returns:
        Processed image (NumPy array, grayscale, upscaled).
    """
    # 0. Initial Upscaling
    if upscale_factor > 1.0:
        height, width = image.shape[:2]
        new_dim = (int(width * upscale_factor), int(height * upscale_factor))
        processed_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)
    else:
        processed_image = image

    # 1. Convert to Grayscale
    if len(processed_image.shape) == 3:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_image

    # 2. Apply Morphological Dilation to potentially thicken text strokes
    # Using a small kernel (e.g., 2x2) for slight effect
    kernel = np.ones((2, 2), np.uint8)
    dilated_gray = cv2.dilate(gray, kernel, iterations=1)

    # 3. Apply slight Gaussian Blur (after dilation)
    blurred_gray = cv2.GaussianBlur(dilated_gray, (3, 3), 0)

    # Return the blurred grayscale image (not binarized)
    return blurred_gray

    # --- Previous complex steps removed again ---
    # Bilateral Filter, CLAHE, Otsu Binarization
