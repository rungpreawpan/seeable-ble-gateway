import os
import uuid

import requests
from flask import Blueprint, jsonify, request
from ultralytics import YOLO

from .custom_function.image_preprocessor import ImagePreprocessor

object_detection = Blueprint("object_detection", __name__)

model = YOLO("yolo11m.pt")
preprocessor = ImagePreprocessor(output_dir="/tmp")
EXPRESS_RESULT_URL = "http://localhost:3000/results"

CONFIDENCE_THRESHOLD = 0.5
BLUR_THRESHOLD = 100


@object_detection.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    processed_path, blur_score = preprocessor.preprocess(filepath)

    if blur_score < BLUR_THRESHOLD:
        return (
            jsonify(
                {
                    "status": "Rejected",
                    "reason": "Image is too blurry for reliable detection.",
                    "blur_score": round(blur_score, 2),
                    "box_count": 0,
                }
            ),
            200,
        )

    results = model(processed_path)
    boxes = []
    image_height, image_width = results[0].orig_shape

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls = int(box.cls[0])
            label = result.names[cls]

            boxes.append(
                {
                    "label": label,
                    "confidence": round(conf, 3),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    payload = {
        "boxes": boxes,
        "image_width": image_width,
        "image_height": image_height,
    }

    try:
        requests.post(EXPRESS_RESULT_URL, json={"objects": payload})
    except Exception as e:
        print("Error sending to server:", e)

    return jsonify(
        {
            "status": "Processed",
            "box_count": len(boxes),
            "blur_score": round(blur_score, 2),
        }
    )