from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np 
import os 
import uuid

app = Flask(__name__)
model = YOLO("yolo_finetune_output/coco128_experiment/weights/best.pt")

@app.route("/predict", methods = ["POST"])
def predict():
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
        result_json = results[0].tojson()

        return jsonify({"predictions": result_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    if __name__ == "__main__":
        app.run(host = "0.0.0.0", port = 8080) # 如果要部署到云端给别人访问，host = 0.0.0.0