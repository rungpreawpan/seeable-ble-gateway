import os, uuid, requests
import numpy as np
import cv2
from flask import Blueprint, request, jsonify
from PIL import Image, ImageDraw, ImageFont

from .midas_model import yolo_model, depth_estimator
from .custom_function.image_preprocessor import ObstacleImagePreprocessor

obstacle_detection = Blueprint("obstacle_detection", __name__)

preprocessor = ObstacleImagePreprocessor(output_dir="/tmp")

EXPRESS_RESULT_URL = "http://localhost:3000/obstacle-detect-results"

PRIORITY_WEIGHTS = {
    "person": 1.00,
    "stairs": 0.95,
    "door": 0.92,
    "elevator": 0.90,
    "chair": 0.85,
    "table": 0.82,
    "cabinet": 0.80,
    "trash bin": 0.76,
    "clutter": 0.75,
    "marker": 0.74,
    "whiteboard": 0.72,
    "exit sign": 0.71,
    "fire extinguisher": 0.69,
    "toilet sign": 0.66,
    "window": 0.57,
}


def get_direction(x_center, image_width):
    ratio = x_center / image_width
    if ratio < 0.33:
        return "ซ้าย"
    elif ratio < 0.66:
        return "หน้า"
    else:
        return "ขวา"


@obstacle_detection.route("/obstacle-detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    processed_path, blur_score = preprocessor.preprocess(filepath)

    results = yolo_model(processed_path)
    image_height, image_width = results[0].orig_shape

    depth = depth_estimator(processed_path)
    depth_map = np.array(depth["predicted_depth"]).astype(np.float32)

    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_map)

    boxes = []
    font = ImageFont.load_default()

    pil_img_only_detect = Image.open(processed_path).convert("RGB")
    detect_draw = ImageDraw.Draw(pil_img_only_detect)

    pil_img = Image.open(processed_path).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    depth_pil = Image.fromarray(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]

            crop_x1 = int(x1 + 0.1 * (x2 - x1))
            crop_y1 = int(y1 + 0.1 * (y2 - y1))
            crop_x2 = int(x2 - 0.1 * (x2 - x1))
            crop_y2 = int(y2 - 0.1 * (y2 - y1))

            roi_depth = depth_norm[crop_y1:crop_y2, crop_x1:crop_x2]
            if roi_depth.size == 0:
                continue

            roi_min, roi_max = roi_depth.min(), roi_depth.max()
            if roi_max - roi_min > 1e-6:
                roi_depth_norm = (roi_depth - roi_min) / (roi_max - roi_min)
            else:
                roi_depth_norm = np.zeros_like(roi_depth)

            median_depth = float(np.median(roi_depth_norm))
            close_depth = float(np.percentile(roi_depth_norm, 10))

            depth_score = 1 - close_depth
            priority = PRIORITY_WEIGHTS.get(label, 0.3) * depth_score

            x_center = (x1 + x2) / 2
            direction = get_direction(x_center, image_width)
            if direction == "หน้า":
                priority *= 1.2
                priority = min(priority, 1.0)

            print(
                f"[DEBUG] {label}: median={median_depth:.2f}, close={close_depth:.2f}, depth_score={depth_score:.2f}, priority={priority:.2f}"
            )

            detect_draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            detect_draw.text(
                (x1, y1 - 12),
                f"{label} {conf:.2f} P:{priority:.2f}",
                fill="yellow",
                font=font,
            )

            depth_diff = abs(median_depth - close_depth)
            if label not in ["bulletin board", "whiteboard"] and depth_diff < 0.05:
                continue
            if priority < 0.4:
                continue

            boxes.append(
                {
                    "label": label,
                    "confidence": round(conf, 3),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "median_depth": round(median_depth, 3),
                    "close_depth": round(close_depth, 3),
                    "direction": direction,
                    "priority": round(priority, 3),
                    "message": f"มี{label} ขวางทางด้าน{direction}",
                }
            )
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            draw.text(
                (x1, y1 - 12),
                f"{label} {conf:.2f} P:{priority:.2f}",
                fill="yellow",
                font=font,
            )

    # visualization
    # depth_filename = filename.replace(".jpg", "_depth.jpg")
    # depth_path = os.path.join("/tmp", depth_filename)
    # depth_pil.save(depth_path)

    # object_detect_priority_path = os.path.join("/tmp", f"{uuid.uuid4().hex}_priority.jpg")
    # pil_img.save(object_detect_priority_path)

    # object_detect_path = os.path.join("/tmp", f"{uuid.uuid4().hex}_detect.jpg")
    # pil_img_only_detect.save(object_detect_path)

    # show images
    # pil_img_only_detect.show()
    # pil_img.show()
    # depth_pil.show()

    payload = {
        "obstacle": boxes,
        "image_width": image_width,
        "image_height": image_height,
    }

    try:
        requests.post(EXPRESS_RESULT_URL, json=payload)
    except Exception as e:
        print("Error sending to server:", e)

    return jsonify({"status": "Processed", "box_count": len(boxes)})
