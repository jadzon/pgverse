import os
import cv2
import numpy as np
import easyocr
import csv
from matplotlib import pyplot as plt
from datetime import datetime

# Initialize the OCR reader - dodanie polskiego języka może pomóc w rozpoznawaniu niektórych znaków
reader = easyocr.Reader(['en', 'pl'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)

def preprocess_image(image):
    """Funkcja do wstępnego przetwarzania obrazu w celu poprawy detekcji tekstu"""
    # Konwersja do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Wyrównanie histogramu dla lepszego kontrastu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Wyostrzenie obrazu
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    
    # Redukcja szumu
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
    
    return denoised

def process_image(image_path, output_dir, csv_writer=None):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Make a copy for highlighting
    highlighted_image = image.copy()
    
    # Preprocessing obrazu dla lepszej detekcji
    processed_image = preprocess_image(image)
    
    # Parametry dla EasyOCR
    min_size = 3  # Minimalna liczba pikseli dla wykrycia tekstu (zmniejszona dla wyższej czułości)
    text_threshold = 0.4  # Próg pewności dla tekstu (zmniejszony)
    low_text = 0.3  # Próg dla filtracji niskiej pewności (zmniejszony)
    link_threshold = 0.3  # Próg dla łączenia tekstu
    
    # Detect text with EasyOCR with adjusted parameters for better detection
    results = reader.readtext(processed_image, paragraph=False, 
                             min_size=min_size, text_threshold=text_threshold, 
                             low_text=low_text, link_threshold=link_threshold,
                             detail=1, batch_size=4, contrast_ths=0.1)
    
    # Prepare data for CSV
    image_name = os.path.basename(image_path)
    category = os.path.basename(os.path.dirname(image_path))
    
    # Funkcja do grupowania wykrytych bloków tekstu w linie
    def group_text_blocks_into_lines(results, y_tolerance=10, x_tolerance=50):
        if not results:
            return []
            
        # Sortuj bloki według współrzędnej y górnego lewego rogu
        sorted_results = sorted(results, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)  # Średnia współrzędna y
        
        lines = []
        current_line = []
        
        for result in sorted_results:
            bbox, text, prob = result
            
            # Filtruj puste lub zbyt krótkie tekstu (mniej niż 1 znak)
            if not text or len(text.strip()) < 1:
                continue
                
            # Weź środkową współrzędną y aktualnego bloku
            current_y = (bbox[0][1] + bbox[2][1]) / 2
            
            # Wysokość obszaru tekstu
            height = bbox[2][1] - bbox[0][1]
            
            # Filtruj zbyt duże obszary (prawdopodobnie nie tekst)
            if height > 100:
                continue
                
            if not current_line:
                # Jeśli linia jest pusta, dodaj pierwszy blok
                current_line.append(result)
            else:
                # Weź średnią współrzędną y ostatniego bloku w aktualnej linii
                last_y = (current_line[-1][0][0][1] + current_line[-1][0][2][1]) / 2
                
                # Dynamiczny próg tolerancji pionowej - zależy od wysokości bloku
                dynamic_y_tolerance = min(y_tolerance, height * 0.7)
                
                # Sprawdź, czy blok jest w tej samej linii (podobna wysokość y)
                if abs(current_y - last_y) < dynamic_y_tolerance:
                    # Sprawdź poziome odległości między blokami
                    last_right_x = max(p[0] for p in current_line[-1][0])  # Najdalej wysunięty punkt w prawo
                    current_left_x = min(p[0] for p in bbox)  # Najbardziej wysunięty punkt w lewo
                    
                    # Dynamiczny próg tolerancji poziomej - zależy od szerokości bloków
                    last_width = max(p[0] for p in current_line[-1][0]) - min(p[0] for p in current_line[-1][0])
                    current_width = max(p[0] for p in bbox) - min(p[0] for p in bbox)
                    dynamic_x_tolerance = min(x_tolerance, max(last_width, current_width) * 0.8)
                    
                    # Jeśli bloki są wystarczająco blisko siebie, dodaj do tej samej linii
                    if current_left_x - last_right_x < dynamic_x_tolerance:
                        current_line.append(result)
                    else:
                        # Jeśli za daleko poziomo, to nowa linia
                        lines.append(current_line)
                        current_line = [result]
                else:
                    # Jeśli blok jest na innej wysokości, rozpocznij nową linię
                    lines.append(current_line)
                    current_line = [result]
        
        # Dodaj ostatnią linię
        if current_line:
            lines.append(current_line)
            
        # Sortuj bloki w każdej linii od lewej do prawej
        for i, line in enumerate(lines):
            lines[i] = sorted(line, key=lambda r: min(p[0] for p in r[0]))
            
        return lines
    
    # Grupuj wykryte bloki tekstu w linie
    lines = group_text_blocks_into_lines(results)
    
    # Highlight text regions for each line
    for line in lines:
        # Połącz bounding box'y w jedną linię
        all_points = []
        line_text = []
        line_prob = []
        
        for bbox, text, prob in line:
            all_points.extend(bbox)
            
            # Usuń niepotrzebne białe znaki
            text = text.strip()
            
            if text:  # Dodaj tylko niepusty tekst
                line_text.append(text)
                line_prob.append(prob)
        
        # Jeśli brak tekstu, pomiń
        if not line_text:
            continue
            
        # Znajdź skrajne punkty dla całej linii
        min_x = min(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_x = max(point[0] for point in all_points)
        max_y = max(point[1] for point in all_points)
        
        # Obliczenie szerokości i wysokości obszaru
        width = max_x - min_x
        height = max_y - min_y
        
        # Pomiń zbyt małe lub zbyt duże obszary - prawdopodobnie nie zawierają tekstu
        if width < 5 or height < 5 or width > 300 or height > 100:
            continue
            
        # Dodaj margines dla lepszego otaczania tekstu
        margin_x = int(width * 0.03)  # 3% marginesu poziomego - zmniejszony
        margin_y = int(height * 0.1)  # 10% marginesu pionowego
        
        min_x = max(0, int(min_x - margin_x))
        min_y = max(0, int(min_y - margin_y))
        max_x = min(image.shape[1], int(max_x + margin_x))
        max_y = min(image.shape[0], int(max_y + margin_y))
        
        # Punkty dla prostokąta
        top_left = (min_x, min_y)
        top_right = (max_x, min_y)
        bottom_right = (max_x, max_y)
        bottom_left = (min_x, max_y)
        
        # Połącz tekst z linii
        full_text = ' '.join(line_text)
        
        # Upewnij się, że tekst jest wart pokazania (przynajmniej jeden znak alfanumeryczny)
        if not any(c.isalnum() for c in full_text):
            continue
            
        # Ogranicz długość tekstu dla czytelności
        if len(full_text) > 30:
            full_text = full_text[:27] + "..."
            
        # Średnia pewność dla całej linii
        avg_prob = sum(line_prob) / len(line_prob)
        
        # Draw the bounding box - zielony dla wysokiej pewności, żółty dla średniej, czerwony dla niskiej
        color = (0, 255, 0)  # Zielony domyślnie
        if avg_prob < 0.7:
            color = (0, 165, 255)  # Pomarańczowy dla średniej pewności
        if avg_prob < 0.5:
            color = (0, 0, 255)  # Czerwony dla niskiej pewności
            
        # Rysuj prostokąt
        cv2.line(highlighted_image, top_left, top_right, color, 2)
        cv2.line(highlighted_image, top_right, bottom_right, color, 2)
        cv2.line(highlighted_image, bottom_right, bottom_left, color, 2)
        cv2.line(highlighted_image, bottom_left, top_left, color, 2)
        
        # Add text label with confidence
        label = f"{full_text} ({avg_prob:.2f})"
        cv2.putText(highlighted_image, label, (top_left[0], top_left[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Print the detected text
        print(f"Detected text in {image_name}: {full_text} (confidence: {avg_prob:.2f})")
        
        # Write to CSV if writer is provided
        if csv_writer:
            csv_writer.writerow([category, image_name, full_text, avg_prob, 
                                 f"{top_left[0]},{top_left[1]},{bottom_right[0]},{bottom_right[1]}"])
    
    # Create output filename
    output_path = os.path.join(output_dir, f"highlighted_{image_name}")
    
    # Save the highlighted image
    cv2.imwrite(output_path, highlighted_image)
    print(f"Saved highlighted image to: {output_path}")
    
    return highlighted_image, results

def main():
    # Define dataset directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'Dataset')
    automatyka_dir = os.path.join(dataset_dir, 'Automatyka')
    elektroniczne_dir = os.path.join(dataset_dir, 'Elektroniczne')
    
    # Create output directories with method subfolder
    output_base = os.path.join(base_dir, 'Output')
    method_dir = os.path.join(output_base, 'EasyOCR')
    automatyka_output = os.path.join(method_dir, 'Automatyka')
    elektroniczne_output = os.path.join(method_dir, 'Elektroniczne')
    
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(method_dir, exist_ok=True)
    os.makedirs(automatyka_output, exist_ok=True)
    os.makedirs(elektroniczne_output, exist_ok=True)
    
    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(method_dir, f"easyocr_results_{timestamp}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Category', 'Image', 'Detected Text', 'Confidence', 'Bounding Box (x1,y1,x2,y2)'])
        
        # Process Automatyka images
        print("\nProcessing Automatyka images...")
        for filename in os.listdir(automatyka_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(automatyka_dir, filename)
                process_image(image_path, automatyka_output, csv_writer)
        
        # Process Elektroniczne images
        print("\nProcessing Elektroniczne images...")
        for filename in os.listdir(elektroniczne_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(elektroniczne_dir, filename)
                process_image(image_path, elektroniczne_output, csv_writer)
    
    print(f"\nAll images processed successfully!")
    print(f"Text extraction results saved to: {csv_path}")

if __name__ == "__main__":
    main()