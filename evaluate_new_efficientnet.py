import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, average_precision_score

def calculate_map(y_true, y_pred, thresholds=[0.5, 0.5]):
    """Calculates mean average precision (mAP) at given thresholds."""
    if len(thresholds) != 2:
        raise ValueError("Please provide two thresholds for mAP calculation.")
    
    ap_scores = []
    for i in range(y_true.shape[1]): # Iterate through each label
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        ap_scores.append(ap)

    # mAP at the first threshold
    map_at_threshold1 = sum(ap_scores) / len(ap_scores) if ap_scores else 0

    # mAP for thresholds between 0.5 and 0.95 with a step of 0.05
    map_05_095 = 0.0
    for t in range(int(thresholds[0] * 100), int(thresholds[1] * 100) + 1, 5):
        threshold = t / 100.0
        temp_ap_scores = []
        for i in range(y_true.shape[1]):
            y_pred_thresholded = (y_pred[:, i] >= threshold).astype(int)
            ap = average_precision_score(y_true[:, i], y_pred_thresholded)
            temp_ap_scores.append(ap)
        map_05_095 += (sum(temp_ap_scores) / len(temp_ap_scores)) if temp_ap_scores else 0

    map_05_095 /= 10  # Divided by the number of thresholds used (10 in this case)

    return map_at_threshold1, map_05_095


def evaluate_classification_model(model_path, test_generator):
    model = load_model(model_path)
    scores = model.evaluate(test_generator)
    y_pred = model.predict(test_generator)
    y_pred_labels = (y_pred > 0.5).astype(int)
    y_true = test_generator.labels

    # Generate classification report 
    class_report = classification_report(y_true, y_pred_labels) 
    print(f"Classification Report for {model_path}:\n{class_report}")

    # Calculate and print mAP
    map_50, map_50_95 = calculate_map(y_true, y_pred, thresholds=[0.5, 0.95])
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}\n")

def main():
    test_data_csv = 'D:/Skripsweet/BISMILLAH_KELAR/EKG_RSAB/test_multilabel.csv'
    test_data_dir = 'D:/Skripsweet/BISMILLAH_KELAR/EKG_RSAB/test/images' 

    df = pd.read_csv(test_data_csv)

    # **Make sure your CSV has a 'filename' column (case-sensitive)**
    print(df.columns)  

    # Construct full paths for filenames (add .png if needed)
    df['filename'] = df['filename'].apply(lambda x: os.path.join(test_data_dir, x + ".jpg"))

    # Check for invalid image filenames
    invalid_files = df[~df['filename'].apply(os.path.exists)]['filename'].tolist()
    if invalid_files:
        print(f"Warning: Found {len(invalid_files)} invalid image filename(s) in CSV. These will be ignored.")
        print(invalid_files)
        df = df[~df['filename'].isin(invalid_files)]

    # Prepare your data generator
    datagen = ImageDataGenerator(rescale=1.0/255)

    # Using full paths from 'filename' column and multiple labels
    test_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename', 
        y_col=df.columns[1:], # Assuming labels start from the 2nd column
        target_size=(640, 640),
        batch_size=10, 
        class_mode='raw',  # Use 'raw' for multi-label 
        shuffle=False
    )

    hdf5_model_dir = 'D:/Skripsweet/BISMILLAH_KELAR/dataset' 
    model_paths = [
        'early_model_ori.h5',
        'early_model_ori_binary.h5',
        'final_model_ori.h5',
        'final_model_ori_binary.h5'
    ]

    for model_name in model_paths:
        model_path = os.path.join(hdf5_model_dir, model_name)
        print(f"Evaluating model: {model_name}")
        evaluate_classification_model(model_path, test_generator)

if __name__ == "__main__":
    main()