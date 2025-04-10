from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import freeze_support


class BlockDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_electrical_symbols(self, image_path, conf_threshold=0.25):
        # Run inference on the image
        results = self.model(image_path, conf=conf_threshold)

        # Plot results with bounding boxes
        res_plotted = results[0].plot()

        # Convert from BGR to RGB for matplotlib
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(res_plotted_rgb)
        plt.axis('off')
        plt.title('Detected Electrical Symbols')

        # Save the output image
        output_path = f"detected_{Path(image_path).name}"
        cv2.imwrite(output_path, res_plotted)
        print(f"Results saved to {output_path}")

        # Print detection details
        boxes = results[0].boxes
        print(f"Found {len(boxes)} electrical components:")

        for box in boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {class_name} ({confidence:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        return results

    def process_circuit_diagrams(self,model_path, images_dir, output_dir="detections", conf_threshold=0.25):
        os.makedirs(output_dir, exist_ok=True)

        # Load the model once
        model = YOLO(model_path)

        # Get all images in the directory
        image_files = list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg"))

        for img_path in image_files:
            print(f"Processing {img_path}")

            # Run inference
            results = model(str(img_path), conf=conf_threshold)

            # Save the result
            res_plotted = results[0].plot()
            output_path = os.path.join(output_dir, f"detected_{img_path.name}")
            cv2.imwrite(output_path, res_plotted)

        print(f"All results saved to {output_dir}")

    def show_plot(self):
        plt.show()

