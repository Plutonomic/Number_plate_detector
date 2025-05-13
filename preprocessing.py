import os
import cv2
import numpy as np
from tqdm import tqdm

# Set paths
RAW_DIR = r'C:\Users\dannb\OneDrive\Documents\Py\Number_plate\dataset\images\train'         # Original images

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Resize to a fixed size (YOLO standard is often 640x640)
    resized = cv2.resize(image, (640, 640))

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization (enhance contrast)
    equalized = cv2.equalizeHist(gray)

    # Apply Gaussian blur (remove noise)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Apply binary thresholding (helps OCR and edge detection)
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def process_all_images():
    images = [f for f in os.listdir(RAW_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

    print(f"🔄 Preprocessing {len(images)} images...")

    for image_file in tqdm(images):
        img_path = os.path.join(RAW_DIR, image_file)
        processed = preprocess_image(img_path)

        if processed is not None:
            print(f"Processing completed")

if __name__ == '__main__':
    process_all_images()
