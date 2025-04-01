from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np 
import os 
import uuid
import io

app = Flask(__name__)
model = YOLO("yolo_finetune_output/coco128_experiment/weights/best.pt")

@app.route("/predict_url", methods = ["POST"])
def predict_from_url():
    try:
        data = request.get_json()
        image_url = data.get("url")

        if not image_url:
            return jsonify({"error": "No URL provided"}), 400

        # get image with request
        import requests
        resp = requests.get(image_url)
        if resp.status_code != 200:
            return jsonify({"error": "Failed to fetch image from URL"}), 400

        # decode image
        image_bytes = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # YOLO inference
        results = model(img, imgsz=640, conf=0.25, iou=0.45)
        img_with_boxes = results[0].plot()

        # return bb
        _, buffer = cv2.imencode(".jpg", img_with_boxes)
        img_bytes = io.BytesIO(buffer)

        return send_file(img_bytes, mimetype="image/jpeg", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods = ["POST"])
def predict_from_local():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error":"No selected file"}), 400 
    
    try:
        # read image(s)
        image_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR) # 把压缩图像decode成nump matrix, with color

        # run inference 
        results = model(img, imgsz = 640, conf = 0.25, iou = 0.45)
        # result_json = results[0].tojson()

        # get result with bb
        img_with_boxes = results[0].plot()
        
        # encode into JPEG 
        _, buffer = cv2.imencode(".jpg", img_with_boxes)
        img_bytes = io.BytesIO(buffer)

        # return jsonify({"predictions": result_json})
        return send_file(img_bytes, mimetype = "image/jpeg", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    if __name__ == "__main__":
        app.run(host = "0.0.0.0", port = 8080) # 如果要部署到云端给别人访问，host = 0.0.0.0