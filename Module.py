# step1. install dataset, and load yolo model
# step2. finetune
# dataset: COCO128; https://ultralytics.com/assets/coco128.zip
# coco 128 has been included in YoloV5 package
# training: 80%, inference: 20%

import os 
import subprocess
import random
import shutil

import torch
from ultralytics import YOLO
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ultralytics.engine.trainer import BaseTrainer

# Create Dic
data_path = "datasets/coco128"
os.makedirs(data_path, exist_ok = True)

# download dataset
if not os.path.exists(f"{data_path}.zip"):
    print("Downloading COCO128 dataset ...")
    subprocess.run(["wget", "https://ultralytics.com/assets/coco128.zip", "-O", f"{data_path}.zip"], check=True)

# decompress COCO128 dataset
if not os.path.exists(f"{data_path}/images/train2017"):
    print("Extracting COCO128 dataset ...")
    subprocess.run(["unzip", f"{data_path}.zip", "-d", "datasets/"], check=True)

train_path = f"{data_path}/images/train2017"
inference_path = f"{data_path}/images/inference"

os.makedirs(inference_path, exist_ok=True)

all_images = [f for f in os.listdir(train_path) if f.endswith(".jpg")]
random.shuffle(all_images)

train_images = all_images[:100]
inference_images = all_images[100:128]

for img in inference_images:
    shutil.move(os.path.join(train_path, img), os.path.join(inference_path, img))

print(f"dataset has been successfully dividened: training {len(train_images)}, inference: {len(inference_images)}")

# create yolo config file 
full_path = os.path.abspath("datasets/coco128")

yaml_content = f"""
path: {full_path}
train: images/train2017/  # training dataset: 100 
val: images/train2017/ # inference dataset: 28
test:  # no test dataset here
nc: 80  # COCO128 80n classes in total 
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
         'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
"""

yaml_path = "datasets/coco128.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)
print(f"coco config file has ben succesfully created. path: {full_path}")

# Fine-tune process
print("starting YOLO Fine-tune ...")
model = YOLO("yolov8n.pt")
model.train(
    data = yaml_path, 
    imgsz = 640,
    epochs = 10, 
    batch = 8, 
    device = "mps" if torch.backends.mps.is_available() else "cpu", 
    # trainer=Customtrainer,
    project = "yolo_finetune_output",
    name = "coco128_experiment"
)
print("YOLOv8 fine-tune completed.")

