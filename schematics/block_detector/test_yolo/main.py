from ultralytics import YOLO
import torch
from PIL import Image
from PIL import ImageChops
from tqdm import tqdm
import os
import sys

#dataset z https://universe.roboflow.com/sarl/electronic-circuits-ifz6c/dataset/1
#dodaj wyeksportuj do tego folderu
model = YOLO("yolov8n.yaml")
result = model.train(data ="data.yaml",epochs = 1)
print(result)


        