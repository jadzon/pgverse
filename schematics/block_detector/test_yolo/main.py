from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import os
import sys
import matplotlib.pyplot as plt
#dataset z https://universe.roboflow.com/sarl/electronic-circuits-ifz6c/dataset/1
#dodaj wyeksportuj do tego folderu
import torch 
torch.cuda.empty_cache()  # clear memory
model = YOLO("yolov8n.pt")
model.to("cuda")
dataset_path = "../dataset"
results = model.train(
    data = f"{dataset_path}/data.yaml",
    epochs = 50,
    imgsz = 640,
    batch = 16,
    name = "single_components",
    device = 1
)
model.export(format = "onnx")

metrics = model.val(data=f"{dataset_path}/data.yaml")
print(metrics.box.map50)
print(metrics.box.map)
plot_results(file = "runs/detect/single_components/results.csv")
plt.savefig("results.png")
        