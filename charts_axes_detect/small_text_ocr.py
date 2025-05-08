import easyocr
import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

# Initialize the reader globally once (can improve performance if called multiple times)
# Consider adding other languages if needed, e.g., ['en', 'pl']
print("Initializing EasyOCR reader for small text detection (language: 'en')...")
easy_reader = easyocr.Reader(['en'], gpu=True)
print("EasyOCR reader initialized.")

# --- Initialize PaddleOCR Reader ---
# Specify language ('en' for English). Set use_gpu=True if CUDA available.
print("Initializing PaddleOCR reader (language: 'en')...")
paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
print("PaddleOCR reader initialized.")

def run_easyocr(image, difficult_mode=False):
    """
    Runs EasyOCR with sensitivity settings. 
    If difficult_mode is True, uses even more sensitive settings.
    """
    try:
        # Base parameters
        params = {
            'detail': 1,
            'paragraph': False,
            'decoder': 'beamsearch',
            'beamWidth': 5,
            'contrast_ths': 0.05,
            'adjust_contrast': 0.1
        }
        
        if difficult_mode:
            # Even more sensitive settings for difficult images
            params.update({
                'mag_ratio': 1.5,  # Larger magnification
                'text_threshold': 0.03,  # Even lower threshold
                'link_threshold': 0.2,   # More aggressive separation
                'ycenter_ths': 0.25,     # More aggressive separation
                'width_ths': 0.3,        # More aggressive separation
                'low_text': 0.15         # More sensitive to weak text signals
            })
            print("  Using difficult mode with higher sensitivity for EasyOCR")
        else:
            # Regular settings
            params.update({
                'mag_ratio': 1.0,
                'text_threshold': 0.05,
                'link_threshold': 0.25,
                'ycenter_ths': 0.35,
                'width_ths': 0.45,
                'low_text': 0.2
            })
        
        return easy_reader.readtext(image, **params)
    except Exception as e:
        print(f"Error during EasyOCR processing: {e}")
        return []

def run_paddleocr(image_path, difficult_mode=False):
    """
    Runs PaddleOCR and formats results.
    If difficult_mode is True, uses more sensitive settings.
    """
    try:
        # PaddleOCR doesn't have as many tunable parameters as EasyOCR
        # But we can use a different approach for difficult images
        if difficult_mode:
            print("  Using difficult mode for PaddleOCR")
            # Read the image and apply additional preprocessing
            image = cv2.imread(image_path)
            if image is None:
                return []
                
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Apply additional contrast enhancement
            # This can help detect faint text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Save temporary enhanced image
            temp_path = image_path + "_temp_enhanced.png"
            cv2.imwrite(temp_path, enhanced)
            
            # Run PaddleOCR on the enhanced image
            result = paddle_reader.ocr(temp_path, cls=True)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
        else:
            # Regular processing
            result = paddle_reader.ocr(image_path, cls=True)
        
        formatted_results = []
        # PaddleOCR returns results in a nested list structure
        if result and result[0]:  # Check if result is not None and first element exists
            for line in result[0]:
                # line format: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, confidence)]
                bbox = line[0]
                text, confidence = line[1]
                # Convert bbox points to float if they aren't already
                float_bbox = [[float(p[0]), float(p[1])] for p in bbox]
                formatted_results.append((float_bbox, text, confidence))
        return formatted_results
    except Exception as e:
        print(f"Error during PaddleOCR processing for {image_path}: {e}")
        return []

def get_bounding_rectangle(box):
    """Converts a 4-point polygon box to an axis-aligned [x_min, y_min, x_max, y_max] rectangle."""
    x_coords = [p[0] for p in box]
    y_coords = [p[1] for p in box]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def is_contained(box1, box2):
    """Checks if box1 is contained within box2 or vice-versa (using axis-aligned rectangles)."""
    rect1 = get_bounding_rectangle(box1)
    rect2 = get_bounding_rectangle(box2)
    
    # Check if rect1 is inside rect2
    cond1 = rect1[0] >= rect2[0] and rect1[1] >= rect2[1] and rect1[2] <= rect2[2] and rect1[3] <= rect2[3]
    # Check if rect2 is inside rect1
    cond2 = rect2[0] >= rect1[0] and rect2[1] >= rect1[1] and rect2[2] <= rect1[2] and rect2[3] <= rect1[3]
    
    return cond1 or cond2

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is expected to be in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
    Uses axis-aligned rectangles derived from the boxes.
    Returns a float between 0 and 1.
    """
    # Convert boxes to x_min, y_min, x_max, y_max format
    box1_coords = get_bounding_rectangle(box1) # Use helper function
    box2_coords = get_bounding_rectangle(box2) # Use helper function
    
    # Calculate intersection area
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No intersection
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_coords[2] - box1_coords[0]) * (box1_coords[3] - box1_coords[1])
    box2_area = (box2_coords[2] - box2_coords[0]) * (box2_coords[3] - box2_coords[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0: # Avoid division by zero if area is zero
        return 0.0
    
    iou = intersection_area / union_area
    # Ensure IoU is clamped between 0 and 1 due to potential floating point inaccuracies
    return max(0.0, min(iou, 1.0))

def text_similarity(text1, text2):
    """
    Calculate a simple text similarity metric.
    Returns True if texts are similar enough, False otherwise.
    More conservative than before.
    """
    # Convert to lowercase for comparison
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    
    # Direct match
    if t1 == t2:
        return True
    
    # Only consider exact matches or if one string fully contains the other
    # and the contained string is at least 80% of the length of the containing string
    if (t1 in t2 and len(t1) >= 0.8 * len(t2)) or (t2 in t1 and len(t2) >= 0.8 * len(t1)):
        return True
    
    # No more fuzzy matching to avoid false positives
    return False

def deduplicate_detections(detections, iou_threshold=0.7):  # Higher threshold (was 0.5)
    """
    Remove duplicate text detections by checking for overlapping boxes and similar text.
    Higher IoU threshold means only very overlapping boxes are considered duplicates.
    
    Args:
        detections: List of (bbox, text, confidence) tuples
        iou_threshold: Minimum IoU threshold to consider boxes as overlapping
        
    Returns:
        List of deduplicated detections, keeping the one with highest confidence when duplicates found
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x[2], reverse=True)
    
    # Initialize result list with the highest confidence detection
    deduplicated = [sorted_detections[0]]
    
    # Check each remaining detection against all accepted ones
    for candidate in sorted_detections[1:]:
        candidate_bbox, candidate_text, _ = candidate
        
        # Flag to determine if this is a duplicate
        is_duplicate = False
        
        for accepted in deduplicated:
            accepted_bbox, accepted_text, _ = accepted
            
            # Calculate IoU between the two boxes
            iou = calculate_iou(candidate_bbox, accepted_bbox)
            
            # Check if boxes overlap significantly and text is similar
            if iou > iou_threshold and text_similarity(candidate_text, accepted_text):
                is_duplicate = True
                break
        
        # If not a duplicate, add to the results
        if not is_duplicate:
            deduplicated.append(candidate)
    
    print(f"  Deduplication: {len(sorted_detections)} → {len(deduplicated)} detections")
    return deduplicated

def calculate_distance(box1, box2):
    """
    Calculate the minimum distance between two bounding boxes.
    Returns 0 if boxes overlap.
    """
    # Convert boxes to x_min, y_min, x_max, y_max format
    def get_box_coordinates(box):
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    box1_coords = get_box_coordinates(box1)
    box2_coords = get_box_coordinates(box2)
    
    # Check if boxes overlap (intersection)
    x_overlap = max(0, min(box1_coords[2], box2_coords[2]) - max(box1_coords[0], box2_coords[0]))
    y_overlap = max(0, min(box1_coords[3], box2_coords[3]) - max(box1_coords[1], box2_coords[1]))
    
    if x_overlap > 0 and y_overlap > 0:
        return 0  # Boxes overlap
    
    # Calculate distances in x and y directions
    if box1_coords[2] < box2_coords[0]:  # box1 is to the left of box2
        x_distance = box2_coords[0] - box1_coords[2]
    elif box2_coords[2] < box1_coords[0]:  # box2 is to the left of box1
        x_distance = box1_coords[0] - box2_coords[2]
    else:  # Boxes overlap in x direction
        x_distance = 0
    
    if box1_coords[3] < box2_coords[1]:  # box1 is above box2
        y_distance = box2_coords[1] - box1_coords[3]
    elif box2_coords[3] < box1_coords[1]:  # box2 is above box1
        y_distance = box1_coords[1] - box2_coords[3]
    else:  # Boxes overlap in y direction
        y_distance = 0
    
    # Return Euclidean distance
    return (x_distance**2 + y_distance**2)**0.5

def merge_boxes(box1, box2):
    """
    Merge two bounding boxes into one that encompasses both.
    """
    all_x = [p[0] for p in box1] + [p[0] for p in box2]
    all_y = [p[1] for p in box1] + [p[1] for p in box2]
    
    x_min, y_min = min(all_x), min(all_y)
    x_max, y_max = max(all_x), max(all_y)
    
    # Create a new box in the format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

def merge_close_detections(detections, min_iou_threshold=0.3):
    """
    Merge text detections that have significant overlap (IoU >= threshold) 
    or where one box is contained within the other.
    
    Args:
        detections: List of (bbox, text, confidence) tuples
        min_iou_threshold: Minimum IoU required to merge blocks (0.0-1.0)
        
    Returns:
        List with merged detections
    """
    if len(detections) <= 1:
        return detections
    
    # Make a copy to avoid modifying the original list
    result = detections.copy()
    
    # Flag to track if any merges happened
    merged = True
    total_merges = 0
    
    # Keep merging until no more merges occur
    while merged:
        merged = False
        i = 0
        
        while i < len(result):
            j = i + 1
            while j < len(result):
                box1, text1, conf1 = result[i]
                box2, text2, conf2 = result[j]
                
                # Calculate IoU
                iou = calculate_iou(box1, box2)
                
                # Check for containment
                contained = is_contained(box1, box2)

                # Merge boxes if IoU is high OR if one box contains the other
                if iou >= min_iou_threshold or contained:
                    # Log why merge happened
                    merge_reason = f"IoU ({iou:.2f} >= {min_iou_threshold})" if iou >= min_iou_threshold else "Containment"
                    # print(f"    Merging box {i} and {j} due to {merge_reason}") # Optional debug log

                    # Merge boxes
                    merged_box = merge_boxes(box1, box2)
                    
                    # Merge text logic (unchanged)
                    if len(text1) > len(text2):
                        merged_text = text1
                    elif len(text2) > len(text1):
                        merged_text = text2
                    else:
                        merged_text = text1 + " " + text2
                    
                    merged_conf = max(conf1, conf2)
                    
                    result[i] = (merged_box, merged_text, merged_conf)
                    result.pop(j)
                    
                    merged = True
                    total_merges += 1
                    # Important: Break inner loop and restart outer loop check after a merge
                    break 
                else:
                    j += 1
            
            if merged:
                # If a merge happened in the inner loop, restart the outer loop from the beginning
                # This ensures that the newly merged block is compared against all others
                break 
            else:
                 # Only increment 'i' if no merge happened for the current 'i'
                i += 1
        # End of inner loops (j)
        # If merged is True here, the while loop continues, restarting the process
        # If merged is False, it means a full pass occurred with no merges, so we exit the while loop
    
    # Final log message (unchanged)
    if total_merges > 0:
        print(f"  Merging overlapping/contained blocks (IoU threshold: {min_iou_threshold}): {len(detections)} → {len(result)} detections, {total_merges} merges performed")
    else:
        print(f"  No overlapping/contained blocks found to merge (IoU threshold: {min_iou_threshold}).")

    return result

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
    if not detections: # Handle empty list case
      return refined
      
    print(f"  Refining {len(detections)} detections (filtering by aspect ratio > {max_aspect_ratio})...")
    filtered_count = 0
    for (bbox, text, confidence) in detections:
        try:
            # Calculate width and height from bbox using the helper
            rect = get_bounding_rectangle(bbox)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            # Avoid division by zero for very thin lines/detections
            if width < 1 or height < 1:
                aspect_ratio = 1 # Treat degenerate boxes as normal
            else:
                aspect_ratio = max(width / height, height / width)

            # Keep the detection only if aspect ratio is within limits
            if aspect_ratio <= max_aspect_ratio:
                refined.append((bbox, text, confidence))
            else:
                # print(f"    Filtering out large/disproportionate box: W={width:.0f}, H={height:.0f}, Ratio={aspect_ratio:.1f}, Text='{text}'") # Optional debug log
                filtered_count += 1
        except Exception as e:
            print(f"    Warning: Error processing bbox {bbox} during refinement: {e}")
            refined.append((bbox, text, confidence)) # Keep if error
            continue
    if filtered_count > 0:
        print(f"    Filtered out {filtered_count} blocks due to aspect ratio.")
    return refined

def detect_text_combined(image_path, min_confidence=0.1, enable_merging=True, iou_merge_threshold=0.3, return_intermediate=False):
    """
    Detects text using both EasyOCR and PaddleOCR, combines the results,
    REFINES based on aspect ratio, removes duplicates, and optionally merges.
    
    If few text blocks are found with regular settings, tries again with more sensitive settings.
    Can optionally return intermediate results for debugging.
    
    Args:
        image_path (str): Path to the preprocessed image file.
        min_confidence (float): Minimum confidence threshold for returning results.
        enable_merging (bool): Whether to merge overlapping text blocks. Default is True.
        iou_merge_threshold (float): Minimum IoU threshold for merging (0.0-1.0). Default is 0.3 (30% overlap).
        return_intermediate (bool): If True, return a dictionary with intermediate results.
        
    Returns:
        list or dict: A processed list of tuples (bbox, text, confidence), or a dictionary 
                      containing 'final', 'combined_raw', 'refined', 'deduplicated', 'merged' results if return_intermediate is True.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return [] if not return_intermediate else {'final': [], 'combined_raw': [], 'refined': [], 'deduplicated': [], 'merged': []}

    # First try with regular settings
    print(f"Running EasyOCR on {image_path}...")
    easy_results_raw = run_easyocr(image, difficult_mode=False)
    easy_results_filtered = [res for res in easy_results_raw if res[2] >= min_confidence]
    print(f"  EasyOCR found {len(easy_results_filtered)} blocks (after conf filter)...")

    print(f"Running PaddleOCR on {image_path}...")
    paddle_results_raw = run_paddleocr(image_path, difficult_mode=False)
    paddle_results_filtered = [res for res in paddle_results_raw if res[2] >= min_confidence]
    print(f"  PaddleOCR found {len(paddle_results_filtered)} blocks (after conf filter)...")

    # Combine results from both engines
    combined_results_before_difficult = easy_results_filtered + paddle_results_filtered
    print(f"  Combined raw results (before difficult mode check): {len(combined_results_before_difficult)} blocks.")
    
    combined_results = combined_results_before_difficult # Start with base results
    
    # If very few results were found, try again with difficult mode
    if len(combined_results) < 3:  # Threshold for triggering difficult mode
        print(f"Few text blocks detected. Trying again with more sensitive settings...")
        
        easy_results_difficult = run_easyocr(image, difficult_mode=True)
        easy_results_difficult_filtered = [res for res in easy_results_difficult if res[2] >= min_confidence * 0.8]
        print(f"  EasyOCR (difficult mode) found {len(easy_results_difficult_filtered)} blocks...")
        
        paddle_results_difficult = run_paddleocr(image_path, difficult_mode=True)
        paddle_results_difficult_filtered = [res for res in paddle_results_difficult if res[2] >= min_confidence * 0.8]
        print(f"  PaddleOCR (difficult mode) found {len(paddle_results_difficult_filtered)} blocks...")
        
        all_combined_results = combined_results + easy_results_difficult_filtered + paddle_results_difficult_filtered
        print(f"  All combined results: {len(all_combined_results)} blocks.")
        
        combined_results = all_combined_results

    # --- Apply ASPECT RATIO REFINEMENT early --- 
    refined_results = refine_detections(combined_results, max_aspect_ratio=5.0)
    refined_output = refined_results.copy() # For intermediate output

    # Store raw combined results *before* refinement for potential return
    combined_raw_output = combined_results.copy() 
    
    # --- Apply deduplication ON REFINED results --- 
    deduplicated_results = deduplicate_detections(refined_results, iou_threshold=0.7)
    print(f"  After deduplication: {len(deduplicated_results)} blocks.")
    deduplicated_output = deduplicated_results.copy()

    # Initialize merged results with deduplicated results
    merged_results = deduplicated_results
    merged_output = merged_results # If merging disabled, this is the final before return
    
    # --- Only merge overlapping detections if enabled ON DEDUPLICATED results --- 
    if enable_merging:
        print(f"  Merging overlapping/contained blocks (IoU threshold: {iou_merge_threshold})...")
        # Pass deduplicated results to merging function
        merged_results = merge_close_detections(deduplicated_results, min_iou_threshold=iou_merge_threshold) 
        print(f"  Results after merging: {len(merged_results)} blocks.")
        merged_output = merged_results.copy()
    else:
        print("  Merging overlapping blocks disabled.")

    final_results = merged_results # Final result is the merged (or just deduplicated if merging disabled)

    if return_intermediate:
        return {
            'final': final_results, # Result after refinement -> deduplication -> merging
            'combined_raw': combined_raw_output, # Before any filtering/dedup/merging
            'refined': refined_output,       # After aspect ratio filter, before dedup/merging
            'deduplicated': deduplicated_output, # After refinement & deduplication, before merging
            'merged': merged_output        # After refinement & deduplication & merging
        }
    else:
        return final_results

if __name__ == "__main__":
    # Test funkcji na przykładowym obrazie
    import sys
    import os
    import json
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Błąd: Plik {image_path} nie istnieje")
            sys.exit(1)
        
        # Uruchom detekcję tekstu
        results = detect_text_combined(
            image_path, 
            min_confidence=0.1, 
            enable_merging=True, 
            iou_merge_threshold=0.4
        )
        
        # Wypisz wyniki
        if results:
            print(f"\nWykryto {len(results)} bloków tekstowych:")
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"{i+1}. '{text}' (pewność: {confidence:.2f})")
            
            # Zapisz wyniki do pliku JSON
            base_name = os.path.splitext(image_path)[0]
            json_path = f"{base_name}_text.json"
            
            # Przygotuj dane do zapisu
            json_data = {
                "image": image_path,
                "blocks": []
            }
            
            for bbox, text, confidence in results:
                # Konwersja bbox na prosty format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                
                block = {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max)
                    }
                }
                json_data["blocks"].append(block)
            
            # Zapisz do pliku JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
                
            print(f"Zapisano wyniki do pliku: {json_path}")
        else:
            print("Nie wykryto żadnego tekstu na obrazie.")
    else:
        print("Użycie: python small_text_ocr.py <ścieżka_do_obrazu>")
        sys.exit(1) 