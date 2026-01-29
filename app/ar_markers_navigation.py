import os
import uuid
from flask import Blueprint, jsonify, request
from ultralytics import YOLO

ar_markers_navigation = Blueprint("ar_markers_navigation", __name__)

model = YOLO("ar-markers-model.pt")
CONFIDENCE_THRESHOLD = 0.5


@ar_markers_navigation.route("/ar-marker", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    results = model(filepath)

    best_marker_id = None
    best_conf = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls = int(box.cls[0])

            if conf > best_conf:
                best_conf = conf
                best_marker_id = cls + 1  

    os.remove(filepath)

    return jsonify(
        {
            "marker_id": best_marker_id,
            "confidence": round(best_conf, 2) if best_marker_id else None,
        }
    )
