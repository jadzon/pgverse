from ultralytics import YOLO
import yolov5
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import freeze_support
import torch

class BlockDetector:
    def __init__(self, circuit_path = "block_detector/models/handwritten.pt",block_path = "block_detector/models/block_diagram.pt" ):
        freeze_support()
        self.circuit_model = YOLO(circuit_path)
        self.diagram_model = yolov5.load(block_path)

    def detect_electrical_symbols(self, image_path, conf_threshold=0.25):
        # Run inference on the image
        results = self.circuit_model(image_path, conf=conf_threshold)

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
        for i,name in enumerate(self.circuit_model.names):
            print(f"  {i}: {self.circuit_model.names[i]}")
        nodes_exist = False
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 12: # Class_id 12 is for nodes
                nodes_exist = True
            class_name = self.circuit_model.names[class_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {class_name} ({confidence:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        return boxes,nodes_exist
    def detect_block_diagrams(self, image_path, conf_threshold=0.25):
        if self.diagram_model is None:
            print("Error: YOLOv5 model not initialized")
            return []
            
        # Run inference using YOLOv5
        # YOLOv5 has a different API
        img = cv2.imread(image_path)
        results = self.diagram_model(img, size=640)
        
        # Get the plotted image from YOLOv5 results
        results.show()

        
        # Print detection details
        print(f"Found {len(results.pred[0])} components:")
        
        # Convert YOLOv5 results to a format similar to your existing code
        boxes = []
        for *xyxy, conf, cls in results.pred[0].cpu().numpy():
            class_id = int(cls)
            class_name = results.names[class_id]
            print(f"  {class_name} ({conf:.2f}) at [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
            
            # Create a Box object similar to YOLOv8/11 for consistency
            box = type('Box', (), {})()
            box.cls = torch.tensor([class_id])
            box.conf = torch.tensor([conf])
            box.xyxy = torch.tensor([[xyxy[0], xyxy[1], xyxy[2], xyxy[3]]])
            boxes.append(box)
            
        return boxes
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

