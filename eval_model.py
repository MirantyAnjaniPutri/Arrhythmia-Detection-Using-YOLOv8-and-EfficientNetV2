from ultralytics import YOLO
import os

# Focus on the box segmentation
def main():
    model1 = YOLO("D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb/yolov8n-obb - nano sgd/weights/best.pt")
    model2 = YOLO("D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb/yolov8n-obb - nano adam/weights/best.pt")
    model3 = YOLO("D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb_SGD/yolov8n-obb/weights/best.pt")
    model4 = YOLO("D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb_Adam/yolov8n-obb/weights/best.pt")

    models_1 = [model1, model2]
    model_names_1 = ["YOLOv8 Nano SGD - Dataset Ori", "YOLOv8 Nano Adam - Dataset Ori"]
    models_2 = [model3, model4]
    model_names_2 = ["YOLOv8 Nano SGD - Dataset Binary", "YOLOv8 Nano Adam - Dataset Binary"]

    # Iterate over each model
    for model, name in zip(models_1, model_names_1):
        # Validate the model
        metrics = model.val(data="D:/Skripsweet/BISMILLAH_KELAR/datasets/data.yaml") 

        # Calculate and print the metrics
        box_precision = metrics.box.map50  # Mean Average Precision at IoU 0.50
        box_recall = metrics.box.mr       # Mean Recall
        box_map50 = metrics.box.map50     # Mean Average Precision at IoU 0.50
        box_map50_95 = metrics.box.map    # Mean Average Precision at IoU 0.50 to 0.95
        
        f1_score = (2 * box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) != 0 else 0
        
        print(f"Results for {name}:")
        print(f"Precision: {round(box_precision, 3)}")
        print(f"Recall: {round(box_recall, 3)}")
        print(f"mAP50: {round(box_map50, 3)}")
        print(f"mAP50-95: {round(box_map50_95, 3)}")
        print(f"F1-score: {round(f1_score, 3)}")

    # Iterate over each model
    for model, name in zip(models_2, model_names_2):
        # Validate the model
        metrics = model.val(data="D:/Skripsweet/BISMILLAH_KELAR/datasets_binarize/data.yaml") 

        # Calculate and print the metrics
        box_precision = metrics.box.map50  # Mean Average Precision at IoU 0.50
        box_recall = metrics.box.mr       # Mean Recall
        box_map50 = metrics.box.map50     # Mean Average Precision at IoU 0.50
        box_map50_95 = metrics.box.map    # Mean Average Precision at IoU 0.50 to 0.95
        
        f1_score = (2 * box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) != 0 else 0
        
        print(f"Results for {name}:")
        print(f"Precision: {round(box_precision, 3)}")
        print(f"Recall: {round(box_recall, 3)}")
        print(f"mAP50: {round(box_map50, 3)}")
        print(f"mAP50-95: {round(box_map50_95, 3)}")
        print(f"F1-score: {round(f1_score, 3)}")        

if __name__ == '__main__':
    main()


# Focus on segmentation
# def main():
#     yolov8s = YOLO("C:/Users/miran/runs/segment/yolov8s-seg5/weights/best.pt")
#     yolov8n = YOLO("C:/Users/miran/runs/segment/yolov8n-seg/weights/best.pt")
#     # yolov8m = YOLO("C:/Users/miran/runs/segment/yolov8m-seg/weights/best.pt")

#     models = [yolov8s, yolov8n] # yolov8m
#     model_names = ["YOLOv8_small", "YOLOv8_nano"] # , "YOLOv8_medium"

#     # Iterate over each model
#     for model, name in zip(models, model_names):
#         # Validate the model
#         metrics = model.val(data="C:/Users/miran/code_skripsi/dataset/data.yaml") 

#         # Calculate and print the metrics
#         precision = metrics.seg.mp
#         recall = metrics.seg.mr
#         f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
#         map50 = metrics.seg.map50
#         map50_95 = round(metrics.seg.map, 3)
#         print(f"Results for {name}:")
#         print(f"mAP50: {round(map50, 3)}")
#         print(f"mAP50-95: {map50_95}")
#         print(f"F1-score: {round(f1_score, 3)}")
#         # print(2*(metrics.seg.p*metrics.seg.p)/(metrics.seg.p+metrics.seg.p))

# if __name__ == '__main__':
#     main()