import os
import uuid

import pytesseract
import requests
from flask import Blueprint, jsonify, request
from PIL import Image

scan_text = Blueprint("scan_text", __name__)

EXPRESS_OCR_RESULT_URL = "http://localhost:3000/ocr-detect"


@scan_text.route("/ocr", methods=["POST"])
def perform_ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    lang = request.form.get("lang", "eng")
    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("/tmp", filename)
    file.save(filepath)

    try:

        img = Image.open(filepath)
        text = pytesseract.image_to_string(img, lang=lang)
        print(f"\n ตรวจจับข้อความด้วย Tesseract ({lang}):\n{text}\n{'='*40}")

        requests.post(EXPRESS_OCR_RESULT_URL, json={"text": text, "lang": lang})

        return jsonify({"status": "Processed", "length": len(text)})
    except Exception as e:
        return jsonify({"error": str(e)})