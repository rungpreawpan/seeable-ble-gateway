import os
import uuid

import cv2
import numpy as np
import requests
from flask import Blueprint, jsonify, request
from ultralytics import YOLO

object_detection = Blueprint("object_detection", __name__)

model = YOLO("yolo11m.pt")
EXPRESS_RESULT_URL = "http://localhost:3000/results"

CONFIDENCE_THRESHOLD = 0.5
BLUR_THRESHOLD = 100


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))  # Resize to standard input size

    # Convert to grayscale to compute brightness and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    print(f"Image brightness: {brightness:.2f}")

    # Adjust brightness using gamma correction
    if brightness < 80:
        gamma = 1.5  # Brighten dark images
    elif brightness > 180:
        gamma = 0.7  # Darken overly bright images
    else:
        gamma = 1.0  # No adjustment needed

    if gamma != 1.0:
        lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]
        ).astype("uint8")
        image = cv2.LUT(image, lut)
        print(f"Applied gamma correction with gamma = {gamma}")

    # Enhance local contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    image = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
    print("Applied CLAHE for contrast enhancement.")

    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)
    print("Applied Gaussian Blur.")

    # Sharpen the image to enhance object edges
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen_kernel)
    print("Applied sharpening filter.")

    # Calculate blur score using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Laplacian variance (blur score): {blur_score:.2f}")

    # Convert to RGB for YOLO
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save preprocessed image
    preprocessed_path = image_path.replace(".jpg", "_pre.jpg")
    cv2.imwrite(preprocessed_path, image)

    return preprocessed_path, blur_score


@object_detection.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    # Run preprocessing and get blur score
    processed_path, blur_score = preprocess_image(filepath)

    # If image is too blurry, skip detection but return status 200
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

    # Run YOLO detection
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