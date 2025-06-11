import os
import cv2
import numpy as np
from pathlib import Path

PROJECT_DIR = "." 
     
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
    noise_map = cv2.absdiff(original_image, filtered_image)
    noisy_pixels = np.sum(noise_map > threshold)
    total_pixels = original_image.size
    noise_percentage = (noisy_pixels / total_pixels) * 100
    return noise_percentage
def process_images_from_folder(input_folder, output_folder, alpha=0.8, background_color=250, max_resolution=1024):
    """
    Processes all images in the input folder and saves the results in the output folder.
    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder where processed images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        if not os.path.isfile(input_file):
            continue
        # Convert .jpg to .png if necessary
        file_name, file_ext = os.path.splitext(filename)
        if file_ext.lower() == ".jpg":
            # Load the .jpg image
            image = cv2.imread(input_file)
            if image is None:
                print(f"Error loading image: {filename}")
                continue
            # Save it as a .png file
            filename = f"{file_name}.png"
            input_file = os.path.join(input_folder, filename)
            cv2.imwrite(input_file, image)
            print(f"Converted {file_name}.jpg to {file_name}.png")
        # Load the image in grayscale
        image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image: {filename}")
            continue
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
        # Apply Bilateral Filter
        filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        # Calculate noise percentage
        noise_percentage = calculate_noise_percentage(image, filtered_image)
        print(f"{filename} - Noise percentage: {noise_percentage:.2f}%")
        # Adjust brightness and contrast
        mean_pixel_value = np.mean(filtered_image)
        beta = 127.5 - (mean_pixel_value * alpha)
        adjusted_image = cv2.convertScaleAbs(filtered_image, alpha=alpha, beta=beta)
        # Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_image = cv2.filter2D(adjusted_image, -1, kernel)
        # Change background color
        _, mask_high = cv2.threshold(sharpened_image, 100, 255, cv2.THRESH_BINARY)
        sharpened_image[mask_high == 255] = background_color
        _, mask_low = cv2.threshold(sharpened_image, 100, 255, cv2.THRESH_BINARY_INV)
        sharpened_image[mask_low == 255] = np.random.randint(0, 51, size=sharpened_image[mask_low == 255].shape)
        # Save the processed image
        output_file = os.path.join(output_folder, filename)
        cv2.imwrite(output_file, sharpened_image)
        print(f"Processed image saved as: {output_file}")

def process_image(image):
    # Make sure the image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized_gray = clahe.apply(gray)
    
    # Noise reduction
    denoised_gray = cv2.fastNlMeansDenoising(equalized_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        denoised_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Enhanced binarization with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    enhanced_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Sobel gradient - useful for text edge detection
    sobelx = cv2.Sobel(denoised_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(denoised_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_edges = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    
    return {
        "original": image,
        "gray": gray,
        "equalized_gray": equalized_gray,
        "denoised_gray": denoised_gray,
        "binary": binary,
        "enhanced_binary": enhanced_binary,
        "edges": sobel_edges
    }

def preprocessing_general(input_folder, output_folder="preprocessed_for_text_detection", debug=False):
    """
    Performs image processing for text detection on all images in a folder.
    
    Args:
        input_folder: Path to the input folder with images
        output_folder: Path to the output folder - default "preprocessed_for_text_detection"
        debug: Whether to save debug files - default False
        
    Returns:
        Number of processed images
    """
    # Function for processing a single image

    
    # Save debug images
    def save_debug(results, debug_folder, image_name):
        # Create subfolder for this image
        image_debug_folder = os.path.join(debug_folder, image_name)
        os.makedirs(image_debug_folder, exist_ok=True)
        
        # Save images in processing order
        cv2.imwrite(os.path.join(image_debug_folder, "01_original.png"), results["original"])
        cv2.imwrite(os.path.join(image_debug_folder, "02_gray.png"), results["gray"])
        cv2.imwrite(os.path.join(image_debug_folder, "03_equalized_gray.png"), results["equalized_gray"])
        cv2.imwrite(os.path.join(image_debug_folder, "04_denoised_gray.png"), results["denoised_gray"])
        cv2.imwrite(os.path.join(image_debug_folder, "05_binary.png"), results["binary"])
        cv2.imwrite(os.path.join(image_debug_folder, "06_enhanced_binary.png"), results["enhanced_binary"])
        cv2.imwrite(os.path.join(image_debug_folder, "07_edges.png"), results["edges"])
    
    # Paths to folders
    full_input_path = os.path.join(PROJECT_DIR, input_folder)
    full_output_path = os.path.join(PROJECT_DIR, output_folder)
    
    # Create output folder
    os.makedirs(full_output_path, exist_ok=True)
    
    # Create debug folder if needed
    debug_folder = None
    if debug:
        debug_folder = os.path.join(full_output_path, "debug")
        os.makedirs(debug_folder, exist_ok=True)
    
    # Load all images from folder and subfolders
    images = []
    folder_path = Path(full_input_path)
    
    print(f"Loading images from folder: {input_folder}")
    
    # Recursive folder search
    for path in folder_path.rglob('*'):
        if path.is_file() and path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    # Relative path to input folder
                    relative_path = path.relative_to(folder_path)
                    images.append((str(relative_path), img))
                else:
                    print(f"Failed to load image: {path}")
            except Exception as e:
                print(f"Error while loading file {path}: {str(e)}")
    
    print(f"Loaded {len(images)} images")
    
    if not images:
        print("No images found. Exiting.")
        return 0
    
    # Image processing
    counter = 0
    output_folder_path = Path(full_output_path)
    
    for relative_path, img in images:
        print(f"Processing image: {relative_path}")
        
        # Generate unique name for debug folder for this image
        path_obj = Path(relative_path)
        file_name = path_obj.stem
        subfolder_name = str(path_obj.parent).replace('\\', '_').replace('/', '_')
        if subfolder_name and subfolder_name != '.':
            debug_name = f"{subfolder_name}_{file_name}"
        else:
            debug_name = file_name
        
        # Process image - use common logic
        results = process_image(img)
        
        # Save debug images if needed
        if debug:
            save_debug(results, debug_folder, debug_name)
        
        # Create appropriate folder structure for final results
        target_folder = output_folder_path / path_obj.parent
        target_folder.mkdir(parents=True, exist_ok=True)
        
        # Save only final result (enhanced binary) to output folder
        save_path = target_folder / f"{file_name}_processed.png"
        cv2.imwrite(str(save_path), results["enhanced_binary"])
        counter += 1
    
    print(f"Saved {counter} processed images in folder: {output_folder}")
    
    if debug:
        print(f"Debug images have been saved in folder: {output_folder}/debug")
    
    return counter
   
def cut_out_blocks(image,coords):
    cut_out_image = image.copy()
    for block in coords:
        x1, y1, x2, y2 = map(int, block.xyxy[0].tolist())
        # Leave a 3-pixel border around blocks to preserve connections
        cv2.rectangle(
            cut_out_image,
            (x1 + 5, y1 + 5),
            (x2 - 5, y2 - 5),
            (255, 255, 255),  # Fill interior with white
            -1
        )
    cv2.imshow("Cut Out Blocks", cut_out_image)
    cv2.waitKey(0)
    return cut_out_image

# if __name__ == "__main__":
#     # Define input folders
#     input_folder1 = os.path.join("..", "text_extraction", "Output", "EasyOCR", "Automatyka")
#     input_folder2 = os.path.join("..", "text_extraction", "Output", "EasyOCR", "Elektroniczne")
#     # Define output folders relative to the current file's directory
#     output_folder1 = os.path.join(".", "Automatyka_Processed")
#     output_folder2 = os.path.join(".", "Elektroniczne_Processed")
#     # Process images from both folders
#     print("Processing images from folder: Automatyka...")
#     process_images_from_folder(input_folder1, output_folder1)
#     print("Processing images from folder: Elektroniczne...")
#     process_images_from_folder(input_folder2, output_folder2)

#for preprocessing_general
# def main():
#     # Example of using the function
#     input_folder = "Dataset"
#     output_folder = "preprocessed_for_xxx"
#     debug = True
    
#     # Process the entire folder
#     preprocessing_general(
#         input_folder=input_folder,
#         output_folder=output_folder,
#         debug=debug
#     )

# if __name__ == "__main__":
#     main() 

