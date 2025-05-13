import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import easyocr
from datetime import datetime

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Streamlit UI Configuration
st.set_page_config(page_title="Number Plate Detection", layout="centered")
st.title("🚗 Number Plate Detection & OCR")
st.markdown("Upload an image of a vehicle to detect the number plate and extract text using OCR.")

# File uploader widget in Streamlit
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image in the input directory
    input_image_path = os.path.join("input", uploaded_file.name)
    os.makedirs("input", exist_ok=True)
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(input_image_path, caption="Uploaded Image", use_column_width=True)

    # Load the trained YOLO model
    model = YOLO(r"C:/Users/dannb/OneDrive/Documents/Py/Number_plate/dataset/best.pt")  # Change path as needed

    # Create output directory with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"C:/Users/dannb/OneDrive/Documents/Py/Number_plate/dataset/images/output/{timestamp}/"
    os.makedirs(save_dir, exist_ok=True)

    # Detect the number plate using the YOLO model
    st.subheader("🔍 Detecting Number Plate...")
    detection_results = model(input_image_path, save=True, conf=0.5)

    # Retrieve the detection result image path
    detected_image_path = os.path.join(save_dir, uploaded_file.name)

    # Check if the detection was successful and display the result
    if os.path.exists(detected_image_path):
        st.image(detected_image_path, caption="Detected Number Plate", use_column_width=True)

        # OCR on the detected number plate
        st.subheader("🔡 Extracted Plate Number (OCR)")
        ocr_result = reader.readtext(detected_image_path)
        extracted_text = ' '.join([text[1] for text in ocr_result])
        
        # Display OCR results
        if extracted_text:
            st.success(f"Detected Text: {extracted_text}")
        else:
            st.warning("No text detected. Try another image.")
    else:
        st.error("Detection failed. Check your model or image quality.")
