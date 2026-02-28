import os
import uuid
from flask import Blueprint, jsonify, request
from ultralytics import YOLO

ar_markers_navigation_test = Blueprint("ar_markers_navigation_test", __name__)

model = YOLO("ar-markers-model.pt")
CONFIDENCE_THRESHOLD = 0.5


@ar_markers_navigation_test.route("/ar-marker-test", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    results = model(filepath)
    image_height, image_width = results[0].orig_shape
    print(image_height, image_width)
    
    boxes = []
    font = ImageFont.load_default()

    best_marker_id = None
    best_conf = 0.4

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
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
