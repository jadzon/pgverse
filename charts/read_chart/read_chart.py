import cv2
import numpy as np

# dane testowe

class ChartReader:
    def __init__(self,chart_img:cv2.Mat,axis_x,axis_y,bbox_x=None,bbox_y=None):
        self.chart_img = chart_img
        self.axis_x = axis_x[0]
        self.axis_y = axis_y[0]
        self.bbox_x = bbox_x
        self.bbox_y = bbox_y
        
    def cut_out_chart(self):
        """
        Cuts out the chart from the image using the provided axis coordinates.
        """
        self.cut_out_image = self.chart_img.copy()
        # Define the region of interest (ROI) using the axis coordinates
        self.offset_x1, self.offset_x2 = int(self.bbox_x["x_min"]), int(self.bbox_x["x_max"])
        self.offset_y1, self.offset_y2 =  int(self.bbox_y["y_min"]),int(self.bbox_x["y_min"])
        # Ensure the coordinates are within the image bounds   
        # Cut out the chart from the image
        self.cut_out_image = self.cut_out_image[self.offset_y1:self.offset_y2, self.offset_x1:self.offset_x2]
        #show image
        cv2.imshow("Cut Out Chart", self.cut_out_image)
        cv2.waitKey(0)
    def skeletonize(self):
        """
        Converts the image to grayscale, applies Gaussian blur, and then performs Zhang.
        """
        self.skeletonized_image = self.cut_out_image.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(self.skeletonized_image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Apply binary thresholding
        _, binary = cv2.threshold(blurred,130, 255, cv2.THRESH_BINARY_INV)
        # Apply Zhang-Suen thinning algorithm
        skeleton = cv2.ximgproc.thinning(binary)
        #show image
        cv2.imshow("Skeleton", skeleton)
        cv2.waitKey(0)
        self.skeletonized_image = skeleton
    def extract_data_points(self):
        """
        Extracts data points from the chart image and converts pixel coordinates to actual values.
        Uses axes interpretation data for accurate conversion.
        
        Args:
            axes_data: Dictionary containing horizontal_axes and vertical_axes data from JSON
            
        Returns:
            List of (x, y) data points representing the chart line
        """
        self.cut_out_chart()
        # Make sure we have the skeletonized image
        if not hasattr(self, 'skeletonized_image'):
            self.skeletonize()
        
        # Find all non-zero pixels (white pixels in the skeleton representing the line)
        points = np.where(self.skeletonized_image > 0)
        # Combine y, x coordinates (points[0] contains y-values, points[1] contains x-values)
        pixel_points = list(zip(points[1], points[0]))  # (x, y) format
        
        # Sort points by x-coordinate to get them in order from left to right
        pixel_points.sort(key=lambda p: p[0])
        
        # Convert pixel coordinates to actual data values
        data_points = []
        x_min_value = self.axis_x["range"]["min"]

        y_min_value = self.axis_y["range"]["min"]
        # Process points, applying the step size
        current_x_pixel = -float('inf')
        for x_pixel, y_pixel in pixel_points:
            # Apply step size (in pixel space)
            if x_pixel - current_x_pixel < self.axis_x["step"]:
                continue
            current_x_pixel = x_pixel
            
            # Check if x_pixel_adjusted is within the image bounds   
            x_pixel_adjusted = x_pixel - self.axis_x["positions"][0] +self.offset_x1
            x_data = (x_min_value+(x_pixel_adjusted / self.axis_x["pixels_per_unit"]))/self.axis_x["scale_factor"] 
            
            # Note: Need to offset y_pixel by the y1 value to align with axis positions
            y_max_pixel = self.cut_out_image.shape[0]
            y_pixel_adjusted = self.offset_y1 + y_max_pixel - self.axis_y["positions"][0] - y_pixel 
            y_data = (y_min_value +  (y_pixel_adjusted / self.axis_y["pixels_per_unit"]))/self.axis_y["scale_factor"]
            
            # Add rounded values to data points
            data_points.append((round(x_data, 2), round(y_data, 2)))
        
        # Display the extracted points
        print(f"Extracted {len(data_points)} data points")
        
        # Visualize the results
        self._visualize_data_points(pixel_points, data_points, self.axis_x, self.axis_y)
        return data_points

    def _map_pixel_to_value(self, pixel, axis_positions, axis_values):
        """
        Maps a pixel coordinate to a value using axis interpretation data.
        Handles both uniform and non-uniform scales through interpolation.
        
        Args:
            pixel: Pixel position to map
            axis_positions: List of pixel positions for reference values
            axis_values: List of values corresponding to axis_positions
            
        Returns:
            Mapped value
        """
        # Handle pixel outside range (extrapolation)
        if pixel <= axis_positions[0]:
            return axis_values[0]
        if pixel >= axis_positions[-1]:
            return axis_values[-1]
        
        # Find the two closest reference points for interpolation
        for i in range(len(axis_positions) - 1):
            if axis_positions[i] <= pixel <= axis_positions[i + 1]:
                # Linear interpolation between the two closest points
                pos1, pos2 = axis_positions[i], axis_positions[i + 1]
                val1, val2 = axis_values[i], axis_values[i + 1]
                
                # Calculate interpolated value
                ratio = (pixel - pos1) / (pos2 - pos1) if pos2 != pos1 else 0
                return val1 + ratio * (val2 - val1)
        
        # Fallback (shouldn't reach here if pixel is in range)
        return None

    def _visualize_data_points(self, pixel_points, data_points, x_axis, y_axis):
        """
        Visualizes extracted data points on the original image.
        
        Args:
            pixel_points: List of pixel coordinates [(x, y), ...]
            data_points: List of data values [(x, y), ...]
            x_axis: X-axis interpretation data
            y_axis: Y-axis interpretation data
        """
        # Create a copy of the original image for visualization
        visualization = self.chart_img.copy()
        
        # Draw all pixel points
        for x_pixel, y_pixel in pixel_points:
            cv2.circle(visualization, (x_pixel + self.offset_x1, y_pixel + self.offset_y1),
                      1, (0, 0, 255), -1)
        
        # Sample some points to label (avoid overcrowding)
        for i, (x_data, y_data) in enumerate(data_points[::max(1, len(data_points)//10)]):
            # Find closest point in pixel_points
            idx = i * max(1, len(data_points)//10)
            x_pixel, y_pixel = pixel_points[idx]
            
            # Adjust coordinates to original image
            x_display = x_pixel + self.offset_x1
            y_display = y_pixel + self.offset_y1
            
            # Draw larger circles at sampled points
            cv2.circle(visualization, (x_display, y_display), 5, (255, 0, 0), -1)
            
            # Add label
            cv2.putText(visualization, f"({x_data}, {y_data})", 
                        (x_display + 5, y_display - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Show visualization
        cv2.imshow("Extracted Data Points", visualization)
        cv2.waitKey(0)

    def _extract_points_basic_scaling(self):
        """
        Legacy method for extracting points using basic scaling when no axes interpretation data is provided.
        """
        # Find all non-zero pixels (white pixels in the skeleton representing the line)
        points = np.where(self.skeletonized_image > 0)
        pixel_points = list(zip(points[1], points[0]))  # (x, y) format
        pixel_points.sort(key=lambda p: p[0])
        
        data_points = []
        
        # Calculate scaling factors
        x_scale = (self.axis_x["range"]["max"] - self.axis_x["range"]["min"]) / self.cut_out_image.shape[1]
        y_scale = (self.axis_y["range"]["max"] - self.axis_y["range"]["min"]) / self.cut_out_image.shape[0]
        
        # Process points
        current_x_pixel = -float('inf')
        for x_pixel, y_pixel in pixel_points:
            if x_pixel - current_x_pixel < self.axis_x["step"]:
                continue
            current_x_pixel = x_pixel
            
            x_data = self.axis_x["range"]["min"] + (x_pixel * x_scale)
            y_data = self.axis_y["range"]["max"] - (y_pixel * y_scale)
            
            data_points.append((round(x_data, 2), round(y_data, 2)))
        
        return data_points