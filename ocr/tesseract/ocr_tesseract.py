import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = 'D:\\nauka\\tesseract z instalatora\\tesseract.exe'

def process_file(file_path, output_txt, psm=1, oem=1):
    """Process a PDF or image file with Tesseract OCR"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    print(f"Processing file: {file_path}")
    
    # Process based on file type
    if file_extension == '.pdf':
        # Convert PDF to images
        try:
            images = convert_from_path(file_path)
            print(f"PDF contains {len(images)} pages")
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return False
    else:
        # For image files, load directly
        try:
            images = [Image.open(file_path)]
            print(f"Loaded image file: {file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    # Process images with Tesseract
    with open(output_txt, "w", encoding="utf-8") as f:
        for i, img in enumerate(images):
            config = f"--psm {psm} --oem {oem}"
            text = pytesseract.image_to_string(img, lang="pol", config=config)
            
            # For single images, don't add page numbering
            if file_extension == '.pdf':
                f.write(f"=== Strona {i+1} ===\n{text}\n\n")
                print(f"Przetworzono stronę {i+1} / {len(images)}")
            else:
                f.write(text)
                print("Przetworzono obraz")
    
    print(f"Wynik zapisany do {output_txt}")
    return True

def main():
    # Example usage
    # You can change these paths and parameters as needed
    file_path = "D:\\nauka\\baza\\ep.pdf"  # Can be PDF or PNG
    output_txt = "wynikep.txt"
    
    # You can also process PNG files:
    # file_path = "D:\\nauka\\baza\\image.png"
    # output_txt = "wynik_png.txt"
    
    process_file(file_path, output_txt, psm=4, oem=3)

if __name__ == "__main__":
    main()