import os
import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        self.image = None
        self.filtered_image = None
        self.sharpened_image = None
        self.adjusted_image = None
        self.background_color = 250  # Default background color
        self.alpha = 0.8  # Default alpha value for brightness and contrast adjustment
        
    def calculate_noise_percentage(self,original_image, filtered_image, threshold=20):
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

    def process_images_from_folder(self,input_folder, output_folder, alpha=0.8, background_color=250, max_resolution=1024):
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
            noise_percentage = self.calculate_noise_percentage(image, filtered_image)
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