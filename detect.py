import argparse
import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

def run_detection(image_path, model_path=r"C:\Users\dannb\OneDrive\Documents\Py\Number_plate\dataset\best.pt", save_dir="outputs"):
    # Load YOLO model
    model = YOLO(model_path)

    # Run inference
    results = model(image_path, save=True, project=save_dir, name="result", exist_ok=True)

    # Get path of saved image with bounding boxes
    result_image_path = os.path.join(save_dir, "result", os.path.basename(image_path))
    
    print(f"\n✅ Detection Complete. Result saved at: {result_image_path}")
    return result_image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv5 detection on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Path to the trained YOLO model")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save detection results")

    args = parser.parse_args()
    run_detection(args.image, args.model, args.output)
