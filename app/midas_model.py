import torch
import numpy as np
from transformers import pipeline
from ultralytics import YOLO
from PIL import Image

yolo_model = YOLO("obstacle-model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
depth_estimator = pipeline(
    "depth-estimation",
    model="Intel/dpt-large",
    device=0 if device == "cuda" else -1
)

dummy = Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8))
_ = depth_estimator(dummy)