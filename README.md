# 🚗 Number Plate Detection and OCR System

This project is a **Number Plate Detection System** that uses **YOLOv8** for real-time object detection and **EasyOCR** to extract the alphanumeric content from number plates. It provides a simple and effective pipeline to detect and read vehicle registration numbers from images using a web interface powered by **Streamlit**.

---

## 📌 Features

* 🔍 Real-time number plate detection using YOLOv8
* 🧠 Text extraction from detected plates using EasyOCR
* 🖼️ Upload and process images through a Streamlit web app
* 📁 Automatically saves processed images with bounding boxes
* ✅ Custom dataset support with YOLO-compatible annotations

---

## 🧠 Technologies Used

* Python 🐍
* YOLOv8 (Ultralytics)
* EasyOCR
* OpenCV
* Streamlit
* LabelImg (for annotation)

---

## 📁 Project Structure

```
Number_plate_detector/
│
├── app.py                  # Streamlit web app
├── detect.py               # Manual detection script
├── preprocessing.py        # Image preprocessing (optional)
├── eda.py                  # Data exploration and visualization
├── train_model.py          # YOLO training script
├── work.py                 # YAML/data configuration script
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── output/         # Detected result images
│   ├── labels/
│   └── best.pt             # Trained YOLOv8 model
│
├── data.yaml               # YOLO training configuration
├── README.md               # Project documentation
└── requirements.txt        # Required Python packages
```

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/number-plate-detector.git
cd number-plate-detector
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR (for EasyOCR)**

* Windows: Download from [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Add the installed path to your system’s environment variables.

4. **Label your own dataset (Optional)**
   Use `LabelImg` and export annotations in YOLO format.

---

## 🚀 Run the Application

### To launch the web app:

```bash
streamlit run app.py
```

### To run detection from CLI:

```bash
python detect.py --image path/to/image.jpg
```

---

## 📊 Sample Output

* Input: Car image with visible license plate
* Output: Bounding box around plate + OCR output
* Example:

  ```
  Detected Text: TN01AB1234
  ```

---

## 📌 Future Enhancements

* Add vehicle make/model detection
* Support for real-time video stream detection
* Export results in CSV/log files
* Integration with traffic surveillance systems

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

---
