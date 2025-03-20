from ultralytics import YOLO
import torch
from PIL import Image
from PIL import ImageChops
from tqdm import tqdm
import os
import sys

model = YOLO("yolov8n.yaml")
result = model.train(data ="data.yaml",epochs = 1)
print(result)


        