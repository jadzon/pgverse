import sys
import os
import cv2 # Added OpenCV import
from pathlib import Path # Added Path import
import json # Added json import for saving results
import numpy as np # Added numpy import for array operations

# -- Path setup for direct script execution --
# current_script_dir is /path/to/pgverse/charts_axes_detect
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root_dir is /path/to/pgverse
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add project_root_dir to sys.path to find 'schematics'
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# Add current_script_dir to sys.path to find 'chart_preprocessing' and 'small_text_ocr' directly
if current_script_dir not in sys.path:
    sys.path.insert(1, current_script_dir) # Insert it after project_root_dir

# Now we can import the TextExtractor using an absolute path from the project root ('pgverse')
# and other modules (chart_preprocessing, small_text_ocr) directly as their directory is also in sys.path.
from schematics.text_extraction.text_extraction_noPreprocess import TextExtractor
from chart_preprocessing import preprocess_for_small_text # Import the new function
from small_text_ocr import detect_text_combined # Import the combined OCR function

# Helper function to draw bounding boxes (similar to TextExtractor._annotate_detected_text)
def annotate_image(image, detected_texts):
    annotated_image = image.copy()
    # If the input image is grayscale, convert it to BGR for colored boxes
    if len(annotated_image.shape) == 2 or annotated_image.shape[2] == 1:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)

    for (bbox, text, confidence) in detected_texts:
        try:
            # Bbox format from OCR: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            # Calculate the axis-aligned bounding rectangle
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))
            
            # Ensure coordinates are within image bounds (optional but good practice)
            h, w = annotated_image.shape[:2]
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)

            # Draw the axis-aligned rectangle
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            # Put text label (optional, can make image busy)
            # label = f"{text} ({confidence:.2f})"
            # cv2.putText(annotated_image, label, (x_min, y_min - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        except Exception as e:
            print(f"Warning: Error drawing bbox {bbox}: {e}")
            continue
    return annotated_image

# Helper function to format results into JSON (similar to TextExtractor)
def format_results_to_json(image_path, detected_texts, image_shape):
    height, width = image_shape[:2]
    result_json = {
        "image_path": str(image_path),
        "blocks": [],
        "image_size": {
            "width": width,
            "height": height
        }
    }
    for bbox, text, confidence in detected_texts:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        block = {
            "coords": [float(x_min), float(y_min), float(x_max), float(y_max)],
            "type": "rectangle", # Default type
            "text": text,
            "confidence": float(confidence)
        }
        result_json["blocks"].append(block)
    return result_json

def refine_detections(detections, max_aspect_ratio=5.0):
    """
    Filters out detections whose bounding boxes have a very large aspect ratio,
    suggesting incorrect grouping of distant elements.

    Args:
        detections (list): List of (bbox, text, confidence) tuples from OCR.
        max_aspect_ratio (float): Maximum allowed ratio of width/height or height/width.

    Returns:
        list: Filtered list of detections.
    """
    refined = []
    for (bbox, text, confidence) in detections:
        try:
            # Calculate width and height from bbox
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)

            # Avoid division by zero for very thin lines/detections
            if width < 1 or height < 1:
                aspect_ratio = 1 # Treat degenerate boxes as normal
            else:
                aspect_ratio = max(width / height, height / width)

            # Keep the detection only if aspect ratio is within limits
            if aspect_ratio <= max_aspect_ratio:
                refined.append((bbox, text, confidence))
            else:
                print(f"Filtering out large/disproportionate box: W={width:.0f}, H={height:.0f}, Ratio={aspect_ratio:.1f}, Text='{text}'")
        except Exception as e:
            print(f"Warning: Error processing bbox {bbox} during refinement: {e}")
            # Optionally keep the box if refinement fails
            # refined.append((bbox, text, confidence))
            continue # Or skip the problematic box
    return refined

def run_chart_text_extraction():
    """
    Applies custom preprocessing, runs combined OCR (EasyOCR + PaddleOCR),
    refines detections, and saves the results.
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(__file__)

    # Define relative paths based on the script's location
    original_input_folder = os.path.join(script_dir, 'charts_examples') # Original images
    preprocessed_folder = os.path.join(script_dir, 'preprocessed_charts') # Folder for custom preprocessed images
    output_folder = os.path.join(script_dir, 'results') # Final output for text extraction

    print(f"Original input folder: {original_input_folder}")
    print(f"Preprocessed images folder: {preprocessed_folder}")
    print(f"Text extraction output folder: {output_folder}")

    # Check if original input folder exists
    if not os.path.isdir(original_input_folder):
        print(f"Error: Original input folder not found at {original_input_folder}")
        return

    # Create output folders if they don't exist
    os.makedirs(preprocessed_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # --- Custom Preprocessing Step ---
    print("Starting custom preprocessing for small text...")
    image_files_original = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files_original.extend(list(Path(original_input_folder).glob(f"*{ext}")))
        image_files_original.extend(list(Path(original_input_folder).glob(f"*{ext.upper()}")))

    processed_count = 0
    for img_path in image_files_original:
        try:
            # Read the original image
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue

            # Apply the custom preprocessing
            processed_image = preprocess_for_small_text(original_image)

            # Construct the path to save the processed image
            base_name = os.path.basename(img_path)
            save_path = os.path.join(preprocessed_folder, base_name)

            # Save the processed image
            cv2.imwrite(save_path, processed_image)
            processed_count += 1

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    print(f"Custom preprocessing complete. {processed_count} images processed and saved in: {preprocessed_folder}")

    if processed_count == 0:
        print("No images were successfully preprocessed. Skipping text extraction.")
        return {}

    # --- Text Extraction Step using Combined OCR ---
    print("\nStarting text extraction using Combined OCR (EasyOCR + PaddleOCR)...")
    all_results = {}
    # Get list of preprocessed images
    image_files_processed = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files_processed.extend(list(Path(preprocessed_folder).glob(f"*{ext}")))
        image_files_processed.extend(list(Path(preprocessed_folder).glob(f"*{ext.upper()}")))
    
    total_text_blocks_final = 0
    for proc_img_path in image_files_processed:
        print(f"\nProcessing {proc_img_path} for text...")
        img_to_annotate = cv2.imread(str(proc_img_path), cv2.IMREAD_UNCHANGED)
        if img_to_annotate is None:
            print(f"Warning: Could not read image {proc_img_path} for annotation, skipping.")
            continue

        base_name = os.path.basename(proc_img_path)
        name, _ = os.path.splitext(base_name)

        # --- Run Combined OCR with Intermediate Results --- 
        detection_results = detect_text_combined(
            str(proc_img_path), 
            min_confidence=0.1, 
            enable_merging=True, 
            iou_merge_threshold=0.4, 
            return_intermediate=True # Request intermediate results
        )
        
        # --- Save Intermediate Annotated Images --- 
        # 1. Raw Combined Results (Before Refinement/Deduplication/Merging)
        if detection_results.get('combined_raw'): # Use .get for safety
            annotated_raw = annotate_image(img_to_annotate, detection_results['combined_raw'])
            raw_path = os.path.join(output_folder, f"{name}_1_raw_combined.png")
            cv2.imwrite(raw_path, annotated_raw)

        # 2. After Refinement (Before Deduplication/Merging)
        if detection_results.get('refined'):
            annotated_refined = annotate_image(img_to_annotate, detection_results['refined'])
            refined_path = os.path.join(output_folder, f"{name}_2_refined.png")
            cv2.imwrite(refined_path, annotated_refined)

        # 3. After Deduplication (Before Merging)
        if detection_results.get('deduplicated'):
            annotated_dedup = annotate_image(img_to_annotate, detection_results['deduplicated'])
            dedup_path = os.path.join(output_folder, f"{name}_3_deduplicated.png")
            cv2.imwrite(dedup_path, annotated_dedup)

        # 4. After Merging (Final result before external refinement - which is now removed)
        if detection_results.get('merged'):
            annotated_merged = annotate_image(img_to_annotate, detection_results['merged'])
            merged_path = os.path.join(output_folder, f"{name}_4_merged.png")
            cv2.imwrite(merged_path, annotated_merged)

        # --- Use the final result directly from detect_text_combined --- 
        # The 'final' key holds the result after internal refinement, deduplication, and merging.
        # NO NEED to call refine_detections externally anymore.
        final_detections = detection_results['final'] 
        total_text_blocks_final += len(final_detections)

        # --- Save Final Annotated Image and JSON --- 
        if final_detections:
            # Use the final detections for the final annotation and JSON
            annotated_final = annotate_image(img_to_annotate, final_detections)
            final_annotated_path = os.path.join(output_folder, f"{name}_5_final.png") # Renamed final step
            cv2.imwrite(final_annotated_path, annotated_final)
            
            json_data = format_results_to_json(proc_img_path, final_detections, img_to_annotate.shape)
            json_path = os.path.join(output_folder, f"{name}_text.json")
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving JSON for {proc_img_path}: {e}")
            all_results[base_name] = json_data 
        else:
            print(f"  No text detected in the final stage for {proc_img_path}.")

    # Final summary - adjust variable name
    print(f"\nText extraction complete for {len(image_files_processed)} images.")
    # print(f"Total text blocks detected by combined OCR (before refinement): {total_text_blocks_before_refine}") # Can remove this intermediate count
    # print(f"Total text blocks after refinement (aspect ratio filter): {total_text_blocks_after_refine}") # Can remove this intermediate count
    print(f"Total final text blocks outputted: {total_text_blocks_final}") # Use the correct final count
    print(f"Annotated images and JSON results saved in: {output_folder}")

    return all_results

if __name__ == "__main__":
    extracted_data = run_chart_text_extraction()
    # You can add further processing of extracted_data here if needed
    # For example, print the text found in the first processed image:
    #     first_image_key = list(extracted_data.keys())[0]
    #     print(f"\nText found in {first_image_key}:")
    #     for block in extracted_data[first_image_key].get('blocks', []):
    #         print(f"- {block['text']} (Confidence: {block['confidence']:.2f})") 