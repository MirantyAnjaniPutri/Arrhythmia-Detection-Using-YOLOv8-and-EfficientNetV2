from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
from ultralytics import YOLO

# Function to plot bounding boxes
def plot_boxes(draw, boxes, labels, color, class_labels):
    for box in boxes:
        cls, coordinates = box

        # Draw polygon (bounding box)
        draw.polygon(coordinates, outline=color, width=3)

        # Add class text
        font = ImageFont.load_default()
        text = class_labels[cls]
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = min(coord[0] for coord in coordinates)
        text_y = min(coord[1] for coord in coordinates) - text_height

        if text_y < 0:
            text_y = 0

        # Draw filled rectangle for text background
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)

        # Draw text on top of the rectangle
        draw.text((text_x, text_y), text, fill='white', font=font)

def display_ground_truths_and_predictions(optimizer, image_path, label_path, class_labels):
    # Determine model directory based on optimizer
    if optimizer.lower() == 'adam':
        model_path = 'D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb/yolov8n-obb - nano adam/weights/best.pt'  # Replace with your actual path
    elif optimizer.lower() == 'sgd':
        model_path = 'D:/Skripsweet/BISMILLAH_KELAR/runs/yolov8n-obb/yolov8n-obb - nano sgd/weights/best.pt'  # Replace with your actual path
    else:
        raise ValueError(f"Optimizer '{optimizer}' is not supported. Choose either 'adam' or 'sgd'.")

    # Load YOLO model
    model = YOLO(model_path)

    # Load image
    image = Image.open(image_path)
    width, height = image.size     
    draw = ImageDraw.Draw(image)

    # Load ground truths
    label_file = label_path  # Assuming label_path directly points to the label file
    if Path(label_file).exists():
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                entries = line.split()
                cls = int(entries[0])
                coordinates = [(int(float(entries[i]) * width), int(float(entries[i+1]) * height)) for i in range(1, len(entries), 2)]
                coordinates.append(coordinates[0])  # Close the polygon
                plot_boxes(draw, [(cls, coordinates)], labels=class_labels, color='red', class_labels=class_labels)

    # Run inference
    results = model(image_path)

    # Plot predictions
    for result in results:
        boxes = result.xyxy
        for box in boxes:
            cls = int(box[5])
            coordinates = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
            plot_boxes(draw, [(cls, coordinates)], labels=class_labels, color='green', class_labels=class_labels)

    # Display the image
    image.show()

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Display ground truths and predictions using YOLO model.")
    parser.add_argument("optimizer", choices=['adam', 'sgd'], help="Type of optimizer used")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("label_path", help="Path to the label file")
    args = parser.parse_args()

    # Class labels for display
    class_labels = ['I arrhythmia', 'II arrhythmia', 'III arrhythmia', 'aVF arrhythmia', 'aVL arrhythmia', 'aVR arrhythmia',
                    'V1 arrhythmia', 'V2 arrhythmia', 'V3 arrhythmia', 'V4 arrhythmia', 'V5 arrhythmia', 'V6 arrhythmia']

    # Call function to display ground truths and predictions
    display_ground_truths_and_predictions(args.optimizer, args.image_path, args.label_path, class_labels)
