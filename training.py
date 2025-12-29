# YOLO Pose Custom Training Script - Testing angle learning capability
from ultralytics import YOLO
import torch
import os

# Path to your dataset config and pretrained model
DATA_YAML = './dataset/version-1/data.yaml'
MODEL = 'yolo11n-pose.pt'  

def main():
    # Detect device (use CUDA if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create YOLO model
    model = YOLO(MODEL)

    # Allow overriding epochs via env var for quick smoke tests
    epochs = int(os.getenv('TRAIN_EPOCHS', '200'))

    # Train the model (pass device explicitly)
    # Pose-specific configuration for better angle learning
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,                      # Number of epochs (override with TRAIN_EPOCHS env var)
        patience=20,                        # Increased patience for angle convergence
        imgsz=1920,                         # Image size matching your data
        batch=8,                            # Increased batch size with A100 GPU
        project='runs/train',               # Output directory
        device=device,
        
        save=True,
        cache=True,                         # Cache images for faster training
        cos_lr=True,                        # Cosine learning rate scheduling
        warmup_epochs=5,                    # Longer warmup for angle stability
        workers=16,                         # Parallel data loading (A100 can handle it)
        seed=42,                            # Reproducible results
        amp=True,                           # Automatic Mixed Precision for memory efficiency

        name='yolo11n-panther_v1-pose-v1',  # Experiment name
    )

    # Print training results
    print(results)


if __name__ == '__main__':
    # Required on Windows when using multiprocessing in DataLoader
    from multiprocessing import freeze_support
    freeze_support()
    main()

