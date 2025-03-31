from ultralytics import YOLO
import os
import argparse

class inference():
    def __init__(self, model_path = "yolo_finetune_output/coco128_experiment/weights/best.pt", output_dir="inference_results"):

        self.model_path= model_path
        # self.inference_dir = inference_dir
        self.output_dir=output_dir

    def yolo_inference(self, inference_dir, imgsz, conf, iou, save, save_txt, save_conf, project, name):
        # load model 
        model = YOLO(self.model_path)
        os.makedirs(self.output_dir, exist_ok=True)

        # inference and save results 
        results = model(
            source=inference_dir,  # single pic / folder
            imgsz=imgsz,
            conf=conf,              
            iou=iou,               
            save=save,              
            save_txt=save_txt,          
            save_conf=save_conf,        
            project=project,     
            name=name,
        )

        print("Inference completed.")

if __name__ == "__main__":
    output_dir="inference_results"
    parse = argparse.ArgumentParser()
    parse.add_argument("--inference_dir", type = str, default=  "datasets/coco128/images/inference") ## 这里是coco128的数据
    parse.add_argument("--imgsz", type = int, default=640)
    parse.add_argument("--conf", type = float, default=0.25)
    parse.add_argument("--iou", type = float, default=0.45)
    parse.add_argument("--save", type=bool, default=True)
    parse.add_argument("--save_txt", type = bool, default=False)
    parse.add_argument("--save_conf", type = bool, default=False)
    parse.add_argument("--project", type = str, default=output_dir)
    parse.add_argument("--name", type = str, default="coco128_inference")
    args = parse.parse_args()

    inference = inference()
    inference.yolo_inference(args.inference_dir, args.imgsz, args.conf, args.iou, args.save, args.save_txt, args.save_conf, args.project, args.name)