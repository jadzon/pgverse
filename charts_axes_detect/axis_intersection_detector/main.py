import cv2
import numpy as np
import math
import os
from typing import List, Tuple, Optional, Dict, Any


def analyze_intersections_near_horizontal_line(
        image_path: str, debug_mode: bool = False
) -> Tuple[Optional[List[Tuple[int, int]]], bool]:
    """
    Analyzes an image to find a main horizontal line, then detects vertical
    segments intersecting a Region of Interest (ROI) around this line.

    Args:
        image_path (str): Path to the input image.
        debug_mode (bool): If True, displays intermediate images, saves them,
                           and provides more verbose output.

    Returns:
        Tuple[Optional[List[Tuple[int, int]]], bool]:
            - points_data (Optional[List[Tuple[int, int]]]):
                - A list of (x, y) tuples if intersections are found.
                - None if no intersections are found or if a critical error occurred.
            - error_flag (bool):
                - False if intersections were successfully found.
                - True if no intersections were found OR if any processing error occurred.
    """
    debug_output_dir = "debug_images"
    if debug_mode:
        print(f"--- Debug Mode Enabled for {image_path} ---")
        if not os.path.exists(debug_output_dir):
            os.makedirs(debug_output_dir)
        print(f"Debug images will be saved in '{debug_output_dir}/'")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Couldn't load the image from '{image_path}'.")
        # No cv2.imshow calls made yet, so no destroyAllWindows needed for this specific error
        return None, True  # Error

    output_image_for_debug = image.copy() if debug_mode else None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if debug_mode:
        cv2.imshow("Debug: Blurred Grayscale", blurred)
        cv2.imwrite(os.path.join(debug_output_dir, "debug_01_blurred.png"), blurred)

    print("Step 1: Detecting the main horizontal line...")
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    if debug_mode:
        cv2.imshow("Debug: Edges", edges)
        cv2.imwrite(os.path.join(debug_output_dir, "debug_02_edges.png"), edges)

    hough_threshold = 50
    min_line_length = int(image.shape[1] * 0.30)
    max_line_gap = 20
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)

    if linesP is None:
        print("No prominent line segments were detected by Hough Transform.")
        if debug_mode: cv2.destroyAllWindows()  # Blurred and Edges might be open
        return None, True  # Error

    candidate_horizontal_lines: List[Dict[str, Any]] = []
    angle_tolerance_degrees = 2.0
    angle_tolerance_radians = angle_tolerance_degrees * np.pi / 180

    for line_segment in linesP:
        x1, y1, x2, y2 = line_segment[0]
        angle = math.atan2(y2 - y1, x2 - x1)
        if abs(angle) < angle_tolerance_radians or \
                abs(abs(angle) - np.pi) < angle_tolerance_radians:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            candidate_horizontal_lines.append({
                'y_avg': (y1 + y2) / 2.0, 'x1': min(x1, x2), 'x2': max(x1, x2), 'length': length
            })

    if not candidate_horizontal_lines:
        print("No suitable horizontal line candidates found after filtering.")
        if debug_mode: cv2.destroyAllWindows()
        return None, True  # Error

    candidate_horizontal_lines.sort(key=lambda l: l['length'], reverse=True)

    main_line_info: Optional[Dict[str, Any]] = None
    ideal_y, ideal_x_start, ideal_x_end = 0, 0, 0
    if candidate_horizontal_lines:  # This check is a bit redundant due to `if not candidate_horizontal_lines` above
        best_line = candidate_horizontal_lines[0]
        main_line_info = {
            'y': best_line['y_avg'], 'x_start': best_line['x1'], 'x_end': best_line['x2']
        }
        ideal_y = int(round(main_line_info['y']))
        ideal_x_start = int(main_line_info['x_start'])
        ideal_x_end = int(main_line_info['x_end'])

        print(f"Main horizontal line identified: Y={ideal_y}, X from {ideal_x_start} to {ideal_x_end}")
        if debug_mode and output_image_for_debug is not None:
            cv2.line(output_image_for_debug, (ideal_x_start, ideal_y), (ideal_x_end, ideal_y), (0, 255, 0), 2)
    # No 'else' needed here as the earlier check for empty candidate_horizontal_lines handles it.

    if debug_mode:
        print("Step 2 (Debug): Creating and showing 1px mask of the main line...")
        line_mask = np.zeros_like(gray)
        cv2.line(line_mask, (ideal_x_start, ideal_y), (ideal_x_end, ideal_y), 255, 1)
        cv2.imshow("Debug: Main Line Mask (1px)", line_mask)
        cv2.imwrite(os.path.join(debug_output_dir, "debug_03_main_line_mask.png"), line_mask)

    print("Step 3: Extracting the Region of Interest (ROI)...")
    roi_height_above_line = 30
    roi_height_below_line = 30
    roi_y_start = max(0, ideal_y - roi_height_above_line)
    roi_y_end = min(gray.shape[0], ideal_y + roi_height_below_line + 1)
    roi_gray_patch = gray[roi_y_start:roi_y_end, ideal_x_start:ideal_x_end]

    if roi_gray_patch.size == 0:
        print(
            f"ERROR: The ROI is empty. y_start={roi_y_start}, y_end={roi_y_end}, x_start={ideal_x_start}, x_end={ideal_x_end}")
        if debug_mode: cv2.destroyAllWindows()
        return None, True  # Error

    if debug_mode:
        print("Step 4 (Debug): Displaying the grayscale ROI.")
        cv2.imshow("Debug: ROI (Grayscale)", roi_gray_patch)
        cv2.imwrite(os.path.join(debug_output_dir, "debug_04_roi_grayscale.png"), roi_gray_patch)

    print("Step 5: Binarizing the ROI...")
    roi_binarization_threshold = 200
    _, roi_binary = cv2.threshold(roi_gray_patch, roi_binarization_threshold, 255, cv2.THRESH_BINARY_INV)

    if debug_mode:
        print(
            f"Debug: Displaying binarized ROI (Threshold={roi_binarization_threshold}). If debug mode is on, ALWAYS inspect this!")
        cv2.imshow(f"Debug: ROI Binarized (Thresh={roi_binarization_threshold})", roi_binary)
        cv2.imwrite(os.path.join(debug_output_dir, "debug_05_roi_binarized.png"), roi_binary)

    print("Step 6: Scanning ROI for vertical intersections...")
    detected_intersections_raw: List[Dict[str, Any]] = []
    min_vertical_segment_length = 4
    max_scan_distance = max(roi_height_above_line, roi_height_below_line)
    min_x_spacing_between_intersections = 3
    line_y_in_roi = ideal_y - roi_y_start
    last_detected_absolute_x = - (min_x_spacing_between_intersections + 1)

    for x_roi in range(roi_binary.shape[1]):
        absolute_x = ideal_x_start + x_roi
        if absolute_x < last_detected_absolute_x + min_x_spacing_between_intersections:
            continue

        segment_found_at_this_x = False
        up_segment_len = 0
        if line_y_in_roi - 1 >= 0:
            y_scan = line_y_in_roi - 1
            while y_scan >= 0 and roi_binary[y_scan, x_roi] == 255 and up_segment_len < max_scan_distance:
                up_segment_len += 1
                y_scan -= 1
        if up_segment_len >= min_vertical_segment_length:
            segment_found_at_this_x = True
            if debug_mode and output_image_for_debug is not None:
                cv2.line(output_image_for_debug, (absolute_x, ideal_y - 1), (absolute_x, ideal_y - up_segment_len),
                         (255, 100, 0), 1)

        down_segment_len = 0
        if line_y_in_roi + 1 < roi_binary.shape[0]:
            y_scan = line_y_in_roi + 1
            while y_scan < roi_binary.shape[0] and roi_binary[
                y_scan, x_roi] == 255 and down_segment_len < max_scan_distance:
                down_segment_len += 1
                y_scan += 1
        if down_segment_len >= min_vertical_segment_length:
            segment_found_at_this_x = True
            if debug_mode and output_image_for_debug is not None:
                cv2.line(output_image_for_debug, (absolute_x, ideal_y + 1), (absolute_x, ideal_y + down_segment_len),
                         (0, 100, 255), 1)

        if segment_found_at_this_x:
            detected_intersections_raw.append(
                {'x': absolute_x, 'y': ideal_y, 'up': up_segment_len, 'down': down_segment_len})
            last_detected_absolute_x = absolute_x

    final_points_to_return: List[Tuple[int, int]] = [(p['x'], p['y']) for p in detected_intersections_raw]

    points_count = len(final_points_to_return)
    print(f"Step 7: Finalizing results. Processed and identified {points_count} intersection point(s).")

    if debug_mode:
        print(f"Debug Mode: Visualizing final results. Points found: {points_count}")
        if output_image_for_debug is not None:
            for point_tuple in final_points_to_return:  # Draw even if it's zero points (loop won't run)
                cv2.circle(output_image_for_debug, point_tuple, 4, (255, 165, 0), -1)
            cv2.imshow("Debug: Final Result - Intersections", output_image_for_debug)
            cv2.imwrite(os.path.join(debug_output_dir, "debug_06_final_output_with_markings.png"),
                        output_image_for_debug)
            print(f"Debug image '{os.path.join(debug_output_dir, 'debug_06_final_output_with_markings.png')}' saved.")

        print("Press any key to close all debug windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not final_points_to_return:  # No points found is considered an error condition
        print("No intersections found or an earlier critical error occurred. Reporting error: True.")
        return None, True
    else:
        # Intersections were found
        return final_points_to_return, False


if __name__ == "__main__":
    image_file = "3.png"
    if not os.path.exists(image_file):
        print(f"image '{image_file}' not found.")
    else:
        points_data, has_error = analyze_intersections_near_horizontal_line(image_path=image_file, debug_mode=False)

        if has_error:
            print(
                f"No points found")
        else:  # No error, points_data should be a list
            print(f"Detected points: {points_data}")
            print(f"Total points found: {len(points_data) if points_data is not None else 'N/A (error occurred)'}")