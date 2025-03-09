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
import torchvision
from ultralytics import YOLO
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2
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

# # add data augmentation
# transform = A.Compose([
#     A.RandomBrightnessContrast(p = 0.2), 
#     A.GaussianBlur(p=0.1), 
#     A.ToGray(p = 0.05), 
#     A.RandomResizedCrop(size = (640,640), scale = (0.5, 1.0), p = 0.3), 
#     ToTensorV2(), 
# ])

# class augmentationYOLO(Dataset):
#     def __init__(self, img_dir, labels_dir, transform = None):
#         self.transform = transform
#         self.img_dir = img_dir
#         self.labels_dir = labels_dir
#         self.image_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    
#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self,index):
#         img_path = os.path.join(self.img_dir, self.image_files[index])
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

#         label_path = os.path.join(self.labels_dir, self.image_files[index].replace(".jpg", ".txt"))
#         with open(label_path, "r") as f:
#             label = f.readlines()

#         if self.transform:
#             augmented = self.transform(image=img)
#             img = augmented["image"]
#             img = torch.tensor(img, dtype=torch.float32)

#         return img, label
    
# class Customtrainer(BaseTrainer):
#     def get_dataloader(self, dataset_path, batch_size=16, mode="train"):
#         return DataLoader(
#             augmentationYOLO(
#                 img_dir=dataset_path, 
#                 labels_dir="datasets/coco128/labels/train2017", 
#                 transform=transform
#             ), 
#             batch_size=batch_size, 
#             shuffle = True
#         )

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
)
print("YOLOv8 fine-tune completed.")

