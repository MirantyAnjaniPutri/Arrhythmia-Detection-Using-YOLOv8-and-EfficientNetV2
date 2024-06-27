import os
import torch
from ultralytics import YOLO

def main():
    # Paths to your YOLOv8 models
    yolo_adam_color = 'D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb/yolov8n-obb - nano adam/weights/best.pt'
    yolo_sgd_color = 'D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb/yolov8n-obb - nano sgd/weights/best.pt'
    yolo_adam_binary = 'D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb_Adam/yolov8n-obb/weights/best.pt'
    yolo_sgd_binary = 'D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb_SGD/yolov8n-obb/weights/best.pt'

    # List of model paths
    model_paths = [yolo_adam_color, yolo_sgd_color, yolo_adam_binary, yolo_sgd_binary]

    # Path to your validation data
    data_path = 'D:/Skripsweet/BISMILLAH_KELAR/EKG_RSAB/data.yaml'

    # Loop through each model path and evaluate the model
    for model_path in model_paths:
        print(f"Evaluating YOLOv8 model: {model_path}")
        
        # Load the model
        model = YOLO(model_path)
        
        # Evaluate the model with reduced batch size
        results = model.val(data=data_path, batch=1)
        
        # Print the evaluation results
        print(f"Results for model {model_path}:")
        print(results)

        # Clear GPU memory
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()