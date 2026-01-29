# #TODO: ต้องรับค่ามาจากทางโทรศัพท์ว่ามีการเคลื่อนไหวด้วยจะได้ไม่แจ้งเตืนซ้ำๆ 

# from flask import Blueprint, request, jsonify
# from ultralytics import YOLO
# import os, uuid, requests

# from .custom_function.image_preprocessor import ImagePreprocessor

# obstacle_warning = Blueprint("obstacle_warning", __name__)

# model = YOLO("yolo11m.pt")
# preprocessor = ImagePreprocessor(output_dir="/tmp")
# EXPRESS_RESULT_URL = "http://localhost:3000/obstacle-detect-results"

# PRIORITY_WEIGHTS = {
#     "person": 1.0,

#     "stairs": 0.95,
#     "escalator": 0.93,
#     "handrail": 0.9,
#     "door": 0.88,
#     "elevator": 0.85,

#     "chair": 0.8,
#     "table": 0.78,
#     "trash bin": 0.75,
#     "cabinet": 0.73,

#     "dog": 0.7,
#     "cat": 0.68,

#     "bulletin board": 0.6,
#     "whiteboard": 0.6,
#     "water dispenser": 0.58,
#     "fire extinguisher": 0.55,
#     "fan": 0.53,
#     "window": 0.5,
#     "exit sign": 0.48,
# }


# def get_direction(x_center, image_width):
#     ratio = x_center / image_width
#     if ratio < 0.33:
#         return "ซ้าย"
#     elif ratio < 0.66:
#         return "หน้า"
#     else:
#         return "ขวา"


# @obstacle_warning.route("/obstacle-detect", methods=["POST"])
# def detect():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]
#     filename = f"{uuid.uuid4().hex}.jpg"
#     filepath = os.path.join("/tmp", filename)
#     file.save(filepath)

#     processed_path, blur_score = preprocessor.preprocess(filepath)
#     results = model(processed_path)
    
#     image_height, image_width = results[0].orig_shape
#     boxes = []

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             conf = float(box.conf[0])
#             cls = int(box.cls[0])
#             label = result.names[cls]

#             box_width = x2 - x1
#             box_height = y2 - y1
#             area = (box_width * box_height) / (image_width * image_height)
#             x_center = (x1 + x2) / 2
#             direction = get_direction(x_center, image_width)
#             priority = PRIORITY_WEIGHTS.get(label, 0.3) * area
            
#             if direction == "หน้า":
#                 priority *= 1.2
#                 priority = min(priority, 1.0)

#             boxes.append(
#                 {
#                     "label": label,
#                     "confidence": round(conf, 3),
#                     "x1": round(x1),
#                     "y1": round(y1),
#                     "x2": round(x2),
#                     "y2": round(y2),
#                     "area": round(area, 4),
#                     "direction": direction,
#                     "priority": round(priority, 3),
#                     "message": f"มี{label} ขวางทางด้าน{direction}",
#                 }
#             )

#     payload = {
#         "obstacle": boxes,
#         "image_width": image_width,
#         "image_height": image_height,
#     }

#     try:
#         requests.post(EXPRESS_RESULT_URL, json=payload)
#     except Exception as e:
#         print("Error sending to server:", e)

#     return jsonify({"status": "Processed", "box_count": len(boxes)})