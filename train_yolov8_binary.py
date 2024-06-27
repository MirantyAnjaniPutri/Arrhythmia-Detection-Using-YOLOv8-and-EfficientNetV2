# OVERALL
import os
import yaml
import itertools
from ultralytics import YOLO

def main():
    # Load dataset configuration
    with open("D:/Skripsweet/BISMILLAH_KELAR/datasets_binarize/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    # Define common training parameters excluding the already trained combination
    param_grid = {
        'model': ['D:/Skripsweet/BISMILLAH_KELAR/yolov8n-obb.pt'],
        'name': ['yolov8n-obb'],  # Add model names to param_grid for combinations
        'data': ['D:/Skripsweet/BISMILLAH_KELAR/datasets_binarize/data.yaml'],
        'epochs': [100],  # Number of epochs to train
        'batch': [32],  # Batch size
        'imgsz': [640],  # Image size
        'optimizer': ['SGD', 'Adam']
    }

    # Generate combinations
    param_combinations = list(itertools.product(*param_grid.values()))

    # Train each remaining combination of parameters
    for params in param_combinations:
        model_path, model_name, data_path, epochs, batch, imgsz, optimizer = params
        
        # Initialize the model with pre-trained weights
        model = YOLO(model_path)
        
        # Define training parameters
        output_dir = f"D:/Skripsweet/BISMILLAH_KELAR/runs/{model_name}_{optimizer}"
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

# ======================================================================================================================================================

# # MODEL 1: Model: yolov8n-obb.pt, Optimizer: SGD
# import os
# import yaml
# from ultralytics import YOLO

# def main():
#     # Load dataset configuration
#     with open("H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml", 'r') as f:
#         data_config = yaml.safe_load(f)

#     # Define training parameters
#     model_path = 'H:/SKRIPSI/BISMILLAH_KELAR/yolov8n-obb.pt'
#     model_name = 'yolov8n-obb'
#     output_dir = f"H:/SKRIPSI/BISMILLAH_KELAR/runs/{model_name}_SGD"
#     train_params = {
#         'data': 'H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml',
#         'epochs': 100,
#         'batch': 8,
#         'imgsz': 640,
#         'optimizer': 'SGD',
#         'name': model_name,
#         'project': output_dir,
#         'device': 0,
#         'save': True,
#         'save_period': 5,
#     }

#     # Initialize the model
#     model = YOLO(model_path)

#     # Check for existing checkpoints
#     checkpoint_path = os.path.join(output_dir, model_name, 'weights', 'last.pt')
#     if os.path.exists(checkpoint_path):
#         train_params['resume'] = checkpoint_path
#         print(f"Resuming training from checkpoint: {checkpoint_path}")
#     else:
#         print(f"Starting new training run for {model_name} with optimizer SGD")

#     # Start training
#     print(f"Training {model_name} with SGD and parameters: {train_params}")
#     model.train(**train_params)
#     print(f"Training for {model_name} with SGD completed. Results saved in {train_params['project']}/{train_params['name']}")

# if __name__ == '__main__':
#     main()

# ======================================================================================================================================================

# MODEL 2: Model: yolov8n-obb.pt, Optimizer: ADAM
# import os
# import yaml
# from ultralytics import YOLO

# def main():
#     # Load dataset configuration
#     with open("H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml", 'r') as f:
#         data_config = yaml.safe_load(f)

#     # Define training parameters
#     model_path = 'H:/SKRIPSI/BISMILLAH_KELAR/yolov8n-obb.pt'
#     model_name = 'yolov8n-obb'
#     output_dir = f"H:/SKRIPSI/BISMILLAH_KELAR/runs/{model_name}_Adam"
#     train_params = {
#         'data': 'H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml',
#         'epochs': 100,
#         'batch': 16,
#         'imgsz': 640,
#         'optimizer': 'Adam',
#         'name': model_name,
#         'project': output_dir,
#         'device': 0,
#         'save': True,
#         'save_period': 5,
#     }

#     # Initialize the model
#     model = YOLO(model_path)

#     # Check for existing checkpoints
#     checkpoint_path = os.path.join(output_dir, model_name, 'weights', 'last.pt')
#     if os.path.exists(checkpoint_path):
#         train_params['resume'] = checkpoint_path
#         print(f"Resuming training from checkpoint: {checkpoint_path}")
#     else:
#         print(f"Starting new training run for {model_name} with optimizer Adam")

#     # Start training
#     print(f"Training {model_name} with Adam and parameters: {train_params}")
#     model.train(**train_params)
#     print(f"Training for {model_name} with Adam completed. Results saved in {train_params['project']}/{train_params['name']}")

# if __name__ == '__main__':
#     main()

# ======================================================================================================================================================

# MODEL 3: Model: yolov8s-obb.pt, Optimizer: SGD
# import os
# import yaml
# from ultralytics import YOLO

# def main():
#     # Load dataset configuration
#     with open("H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml", 'r') as f:
#         data_config = yaml.safe_load(f)

#     # Define training parameters
#     model_path = 'H:/SKRIPSI/BISMILLAH_KELAR/yolov8s-obb.pt'
#     model_name = 'yolov8s-obb'
#     output_dir = f"H:/SKRIPSI/BISMILLAH_KELAR/runs/{model_name}_SGD"
#     train_params = {
#         'data': 'H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml',
#         'epochs': 100,
#         'batch': 16,
#         'imgsz': 640,
#         'optimizer': 'SGD',
#         'name': model_name,
#         'project': output_dir,
#         'device': 0,
#         'save': True,
#         'save_period': 5,
#     }

#     # Initialize the model
#     model = YOLO(model_path)

#     # Check for existing checkpoints
#     checkpoint_path = os.path.join(output_dir, model_name, 'weights', 'last.pt')
#     if os.path.exists(checkpoint_path):
#         train_params['resume'] = checkpoint_path
#         print(f"Resuming training from checkpoint: {checkpoint_path}")
#     else:
#         print(f"Starting new training run for {model_name} with optimizer SGD")

#     # Start training
#     print(f"Training {model_name} with SGD and parameters: {train_params}")
#     model.train(**train_params)
#     print(f"Training for {model_name} with SGD completed. Results saved in {train_params['project']}/{train_params['name']}")

# if __name__ == '__main__':
#     main()

# ======================================================================================================================================================

# MODEL 4: Model: yolov8s-obb.pt, Optimizer: ADAM
# import os
# import yaml
# from ultralytics import YOLO

# def main():
#     # Load dataset configuration
#     with open("H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml", 'r') as f:
#         data_config = yaml.safe_load(f)

#     # Define training parameters
#     model_path = 'H:/SKRIPSI/BISMILLAH_KELAR/yolov8s-obb.pt'
#     model_name = 'yolov8s-obb'
#     output_dir = f"H:/SKRIPSI/BISMILLAH_KELAR/runs/{model_name}_Adam"
#     train_params = {
#         'data': 'H:/SKRIPSI/BISMILLAH_KELAR/datasets_binarize/data.yaml',
#         'epochs': 100,
#         'batch': 16,
#         'imgsz': 640,
#         'optimizer': 'Adam',
#         'name': model_name,
#         'project': output_dir,
#         'device': 0,
#         'save': True,
#         'save_period': 5,
#     }

#     # Initialize the model
#     model = YOLO(model_path)

#     # Check for existing checkpoints
#     checkpoint_path = os.path.join(output_dir, model_name, 'weights', 'last.pt')
#     if os.path.exists(checkpoint_path):
#         train_params['resume'] = checkpoint_path
#         print(f"Resuming training from checkpoint: {checkpoint_path}")
#     else:
#         print(f"Starting new training run for {model_name} with optimizer Adam")

#     # Start training
#     print(f"Training {model_name} with Adam and parameters: {train_params}")
#     model.train(**train_params)
#     print(f"Training for {model_name} with Adam completed. Results saved in {train_params['project']}/{train_params['name']}")

# if __name__ == '__main__':
#     main()