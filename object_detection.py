from flask import Flask, request, jsonify
from ultralytics import YOLO
import os, uuid, requests

app = Flask(__name__)
model = YOLO('yolov8n.pt')
EXPRESS_RESULT_URL = 'http://localhost:3000/results'

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)

    results = model(filepath)
    boxes = []

    image_height, image_width = results[0].orig_shape

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]

            boxes.append({
                "label": label,
                "confidence": round(conf, 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

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

if __name__ == '__main__':
    app.run(port=5001)