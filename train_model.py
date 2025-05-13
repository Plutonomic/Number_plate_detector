from ultralytics import YOLO
import os

# Set paths
DATA_YAML = r'C:\Users\dannb\OneDrive\Documents\Py\Number_plate\dataset\images\data.yaml'  # This YAML file defines train/val image paths and class info
MODEL = 'yolov5s.pt'     # You can also try yolov5n.pt (nano), yolov5m.pt, etc.
EPOCHS = 2
BATCH = 16
IMG_SIZE = 640

def train_model():
    # Load the model
    model = YOLO(MODEL)

    # Train the model
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project='runs/train',
        name='number_plate_detector',
        exist_ok=True
    )

    print("✅ Training completed!")

if __name__ == '__main__':
    train_model()
