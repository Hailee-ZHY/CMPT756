from flask import Flask, request, jsonify, send_file, send_file
from ultralytics import YOLO
import cv2
import numpy as np 
import os
import uuid
import io
import shutil
from datetime import datetime
import zipfile
from urllib.parse import urlparse
import requests

app = Flask(__name__)
model = YOLO("yolo_finetune_output/coco128_experiment/weights/best.pt")

@app.route("/predict_url", methods = ["POST"])
def predict_from_url():
    try:
        data = request.get_json()
        urls = data.get("url")

        if not url or not isinstance(urls, list):
            return jsonify({"error": "No URL provided"}), 400

        # Prepare temp output folder
        temp_folder = f"/tmp/yolo_url_output_{uuid.uuid4().hex}"
        os.makedirs(temp_folder, exist_ok=True)

        processed_files = []
        failed_urls = []

        for url in urls:
            try:
                resp = requests.get(url)
                if resp.status_code != 200:
                    failed_urls.append(url)
                    continue
                # decode image
                image_bytes = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    failed_urls.append(url)
                    continue
                
                # YOLO inference
                results = model(img, imgsz=640, conf=0.25, iou=0.45)
                img_with_boxes = results[0].plot()

                # Try to extract filename from URL
                parsed = urlparse(url)
                filename = os.path.basename(parsed.path)
                if not filename or "." not in filename:
                    filename = f"img_{uuid.uuid4().hex[:6]}.jpg"

                save_path = os.path.join(temp_folder, filename)
                cv2.imwrite(save_path, img_with_boxes)
                processed_files.append(filename)

            except Exception:
                failed_urls.append(url)

        if not processed_files:
            shutil.rmtree(temp_folder)
            return jsonify({"error": "All URLs failed", "failed_urls": failed_urls}), 500
        
        # create zip
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"/tmp/yolo_url_results_{timestamp}.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for filename in processed_files:
                file_path = os.path.join(temp_folder, filename)
                zipf.write(file_path, arcname=filename)

        shutil.rmtree(temp_folder)

        return_data = None
        with open(zip_filename, 'rb') as f:
            return_data = io.BytesIO(f.read())

        os.remove(zip_filename)
        return send_file(return_data, mimetype='application/zip',
                         as_attachment=True, download_name=f"yolo_url_results_{timestamp}.zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods = ["POST"])
def predict_from_local():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error":"No selected file"}), 400 
    
    # create temp folder, which will be deleted after inefered
    temp_folder = f"/tmp/yolo_output_{uuid.uuid4().hex}"
    os.makedirs(temp_folder, exist_ok=True)

    process_files = []
    failed_files = []
    
    for file in files:
        try:
            # read image(s)
            image_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR) # 把压缩图像decode成nump matrix, with color
            if img is None:
                failed_files.append(file.filename)
                continue

            # run inference 
            results = model(img, imgsz = 640, conf = 0.25, iou = 0.45)
            # get result with bb
            img_with_boxes = results[0].plot()
        
            # save img
            save_path = os.path.join(temp_folder, file.filename)
            cv2.imwrite(save_path, img_with_boxes)
            process_files.append(file.filename)

        except Exception:
            failed_files.append(file.filename)

    # Nothing processed
    if not process_files:
        shutil.rmtree(temp_folder)
        return jsonify({"error": "All images failed to process", "failed_files":failed_files}), 500
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"/tmp/yolo_results_{timestamp}.zip"

    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for filename in process_files:
            file_path = os.path.join(temp_folder, filename)
            zipf.write(file_path, arcname = filename)

    # clean up temp folder after zipping
    shutil.rmtree(temp_folder)

    # send zip to user-end and remove it
    return_data = None
    with open(zip_filename, "rb") as f:
        return_data = io.BytesIO(f.read())

    os.remove(zip_filename)
    return send_file(return_data, mimetype='application/zip', as_attachment=True, download_name=f"yolo_results_{timestamp}.zip")

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080) # 如果要部署到云端给别人访问，host = 0.0.0.0