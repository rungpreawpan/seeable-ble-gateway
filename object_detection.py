# import os
# import uuid

# import requests
# from flask import Flask, jsonify, request
# from ultralytics import YOLO

# app = Flask(__name__)
# model = YOLO("yolo11m.pt")
# EXPRESS_RESULT_URL = "http://localhost:3000/results"

# CONFIDENCE_THRESHOLD = 0.5


# @app.route("/detect", methods=["POST"])
# def detect():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]
#     filename = f"{uuid.uuid4().hex}.jpg"
#     filepath = os.path.join("/tmp", filename)
#     file.save(filepath)

#     results = model(filepath)
#     boxes = []

#     image_height, image_width = results[0].orig_shape

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             conf = float(box.conf[0])
#             if conf < CONFIDENCE_THRESHOLD:
#                 continue

#             cls = int(box.cls[0])
#             label = result.names[cls]

#             boxes.append(
#                 {
#                     "label": label,
#                     "confidence": round(conf, 3),
#                     "x1": x1,
#                     "y1": y1,
#                     "x2": x2,
#                     "y2": y2,
#                 }
#             )

#     payload = {
#         "boxes": boxes,
#         "image_width": image_width,
#         "image_height": image_height,
#     }

#     try:
#         requests.post(EXPRESS_RESULT_URL, json={"objects": payload})
#     except Exception as e:
#         print("Error sending to server:", e)

#     return jsonify({"status": "Processed", "box_count": len(payload)})


# if __name__ == "__main__":
#     app.run(port=5001, debug=True)


import os
import uuid

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolo11m.pt")
EXPRESS_RESULT_URL = "http://localhost:3000/results"

CONFIDENCE_THRESHOLD = 0.5


# def preprocess_image(image_path):
#     image = cv2.imread(image_path)


#     image = cv2.resize(image, (640, 640))
#     image = cv2.GaussianBlur(image, (3, 3), 0)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.equalizeHist(
#         cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     )

#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     preprocessed_path = image_path.replace(".jpg", "_pre.jpg")
#     cv2.imwrite(preprocessed_path, image)
#     return preprocessed_path


def preprocess_image(image_path):
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    print(f"Image brightness: {brightness}")

    if brightness < 80:
        gamma = 1.5
        look_up_table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        image = cv2.LUT(image, look_up_table)
        print("Applied gamma correction to brighten image.")

    elif brightness > 180:
        gamma = 0.7
        look_up_table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        image = cv2.LUT(image, look_up_table)
        print("Applied gamma correction to darken image.")

    image = cv2.GaussianBlur(image, (3, 3), 0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    preprocessed_path = image_path.replace(".jpg", "_pre.jpg")
    cv2.imwrite(preprocessed_path, image)
    return preprocessed_path


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    processed_path = preprocess_image(filepath)

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

    return jsonify({"status": "Processed", "box_count": len(payload)})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
