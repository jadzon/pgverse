import cv2
import os
from typing import Dict
import logging

class DebugUtils:
    """Klasa pomocnicza do obsługi debugowania i zapisywania obrazów."""
    
    # Kolejność etapów przetwarzania
    PROCESSING_STEPS = [
        'original',
        'gray',
        'masked_chars',
        'binary',
        'denoised',
        'edges',
        'dilated',
        'connected'
    ]
    
    @staticmethod
    def save_debug_images(debug_images: Dict, output_dir: str, filename: str):
        """Zapisuje obrazy debugowania do katalogu."""
        try:
            logging.info(f"Rozpoczynam zapisywanie obrazów debugowych dla {filename}")
            logging.info(f"Liczba dostępnych obrazów: {len(debug_images)}")
            logging.info(f"Katalog wyjściowy: {output_dir}")
            
            # Zapisz obrazy w kolejności przetwarzania
            for idx, step in enumerate(DebugUtils.PROCESSING_STEPS, 1):
                if step in debug_images and debug_images[step] is not None:
                    logging.info(f"Przetwarzanie obrazu dla etapu: {step}")
                    logging.info(f"Rozmiar obrazu: {debug_images[step].shape}")
                    logging.info(f"Typ danych obrazu: {debug_images[step].dtype}")
                    
                    # Dodaj opis etapu na obrazie
                    img_with_text = debug_images[step].copy()
                    if len(img_with_text.shape) == 2:
                        img_with_text = cv2.cvtColor(img_with_text, cv2.COLOR_GRAY2BGR)
                        logging.info(f"Konwersja do koloru dla etapu: {step}")
                    
                    # Dla obrazu detected_text, zmień kolor na czerwony i dodaj ramkę
                    if step == 'detected_text':
                        # Utwórz maskę dla niezerowych pikseli
                        mask = img_with_text[:, :, 0] > 0
                        # Zmień kolor na czerwony tylko dla niezerowych pikseli
                        img_with_text[mask] = [0, 0, 255]  # BGR format
                        
                        # Dodaj ramkę wokół wykrytych obszarów
                        contours, _ = cv2.findContours(debug_images[step], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(img_with_text, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Dodaj tekst opisujący etap
                    cv2.putText(img_with_text, f"Step: {step}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Zapisz obraz z numerem sekwencyjnym
                    output_path = os.path.join(output_dir, f"{filename}_{idx:02d}_{step}.png")
                    logging.info(f"Zapisywanie do: {output_path}")
                    success = cv2.imwrite(output_path, img_with_text)
                    if not success:
                        logging.error(f"Błąd podczas zapisywania obrazu: {output_path}")
                    else:
                        logging.info(f"Pomyślnie zapisano obraz: {output_path}")
                else:
                    logging.warning(f"Brak obrazu dla etapu: {step}")
                    
        except Exception as e:
            logging.error(f"Błąd podczas zapisywania obrazów debugowych: {str(e)}")
            import traceback
            logging.error(traceback.format_exc()) 