import os
import requests

url = "http://localhost:8080/predict" # service address
image_dir = "datasets/coco128/images/inference" # 存放测试文件的目录

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    with open(img_path, "rb") as f:
        reponse = requests.post(url, files={"file":f})
    
    print(f"Image:{img_name}")
    print("Response:", reponse.json())
    print("-"*40)