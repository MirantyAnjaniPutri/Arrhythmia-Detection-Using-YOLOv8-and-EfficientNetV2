import yaml
import itertools
from ultralytics import YOLO
import os

def main():
    # Load dataset configuration
    with open("D:/Skripsweet/BISMILLAH_KELAR/datasets/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    # Define common training parameters excluding the already trained combination
    param_grid = {
        'model': ['D:/Skripsweet/BISMILLAH_KELAR/yolov8n-obb.pt', 'D:/Skripsweet/BISMILLAH_KELAR/yolov8s-obb.pt'],
        'name': ['yolov8n-obb', 'yolov8s-obb'],  # Add model names to param_grid for combinations
        'data': ['D:/Skripsweet/BISMILLAH_KELAR/datasets/data.yaml'],
        'epochs': [100],  # Number of epochs to train
        'batch': [32],  # Batch size
        'imgsz': [640],  # Image size
        'optimizer': ['SGD', 'Adam']
    }

    # Generate combinations
    param_combinations = list(itertools.product(*param_grid.values()))

    # Filter out the combination that has already been trained (yolov8n-obb with optimizer SGD)
    trained_combination = ('D:/Skripsweet/BISMILLAH_KELAR/yolov8n-obb.pt', 'yolov8n-obb', 'D:/Skripsweet/BISMILLAH_KELAR/datasets/data.yaml', 100, 32, 640, 'SGD')
    remaining_combinations = [comb for comb in param_combinations if comb != trained_combination]

    # Train each remaining combination of parameters
    for params in remaining_combinations:
        model_path, model_name, data_path, epochs, batch, imgsz, optimizer = params
        
        # Initialize the model with pre-trained weights
        model = YOLO(model_path)
        
        # Define training parameters
        output_dir = f"D:/Skripsweet/BISMILLAH_KELAR/runs/{model_name}"
        train_params = {
            'data': data_path,
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'optimizer': optimizer,
            'name': model_name,  # Unique name for each run based on the model
            'project': output_dir,  # Output directory
            'device': 0,  # Use the first GPU
            'save': True,
            'save_period': 5,  # Save checkpoint every epoch
        }

        # Check for existing checkpoints
        checkpoint_path = os.path.join(output_dir, model_name, 'weights', 'last.pt')
        if os.path.exists(checkpoint_path):
            train_params['resume'] = checkpoint_path  # Resume from the last checkpoint if it exists
            print(f"Resuming training from checkpoint: {checkpoint_path}")
        else:
            print(f"Starting new training run for {model_name} with optimizer {optimizer}")

        # Start training
        print(f"Training {model_name} with optimizer {optimizer} and parameters: {train_params}")
        model.train(**train_params)
        print(f"Training for {model_name} with optimizer {optimizer} completed. Results saved in {train_params['project']}/{train_params['name']}")

    print("All remaining models have been trained.")

if __name__ == '__main__':
    main()
