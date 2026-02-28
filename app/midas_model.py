# import torch
# import numpy as np
# from transformers import pipeline
# from ultralytics import YOLO
# from PIL import Image

# yolo_model = YOLO("obstacle-model.pt")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# depth_estimator = pipeline(
#     "depth-estimation",
#     model="Intel/dpt-large",
#     device=0 if device == "cuda" else -1
# )

# dummy = Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8))
# _ = depth_estimator(dummy)

import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor
from ultralytics import YOLO
from PIL import Image

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

yolo_model = YOLO("obstacle-model.pt")
yolo_model.to("cpu")

DEPTH_MODEL_NAME = "Intel/dpt-hybrid-midas"

processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
depth_model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)

depth_model.to(device)
depth_model.eval()


def run_depth(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = predicted_depth.squeeze().cpu().numpy().astype(np.float32)
    return depth


dummy = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))

_ = yolo_model(dummy)
_ = run_depth(dummy)
