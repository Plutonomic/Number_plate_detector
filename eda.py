import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dataset paths
IMAGE_DIR = r"C:\Users\dannb\OneDrive\Documents\Py\Number_plate\dataset\images\train"
LABEL_DIR = r"C:\Users\dannb\OneDrive\Documents\Py\Number_plate\dataset\yolo_labels"

def load_image_paths(image_dir):
    return [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.jpg', '.png', '.jpeg'))]

def load_labels(label_dir):
    return [os.path.join(label_dir, file) for file in os.listdir(label_dir) if file.endswith('.txt')]

def check_image_sizes(image_paths):
    sizes = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((w, h))
    return sizes

def analyze_label_distribution(label_paths):
    label_counts = {}
    for label_file in label_paths:
        with open(label_file, 'r') as file:
            for line in file.readlines():
                class_id = line.split()[0]
                label_counts[class_id] = label_counts.get(class_id, 0) + 1
    return label_counts

def plot_image_size_distribution(sizes):
    widths, heights = zip(*sizes)
    plt.figure(figsize=(8, 6))
    sns.histplot(widths, color='blue', kde=True, label='Width')
    sns.histplot(heights, color='orange', kde=True, label='Height')
    plt.title("Image Width & Height Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_class_distribution(label_counts):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
    plt.title("Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def main():
    print("📊 Starting EDA...")

    image_paths = load_image_paths(IMAGE_DIR)
    label_paths = load_labels(LABEL_DIR)

    print(f"Total images: {len(image_paths)}")
    print(f"Total label files: {len(label_paths)}")

    sizes = check_image_sizes(image_paths)
    label_counts = analyze_label_distribution(label_paths)

    plot_image_size_distribution(sizes)
    plot_class_distribution(label_counts)

    print("✅ EDA Completed.")

if __name__ == "__main__":
    main()
