import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pytesseract
import fitz  # PyMuPDF

def pdf_to_images(pdf_file):
    """Convert PDF to a list of images, one per page."""
    pdf_data = pdf_file.read()
    doc = fitz.open("pdf", pdf_data)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def preprocess_image(image):
    """Basic preprocessing for OCR."""
    # Convert PIL image to numpy array
    image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale Image", use_column_width=True)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.image(binary, caption="Binarized Image (Otsu's Thresholding)", use_column_width=True)
    
    return binary

def perform_ocr(image):
    """Perform OCR on the given image."""
    return pytesseract.image_to_string(image, lang='eng', config='--psm 3')

# Streamlit UI
st.header("OCR on PDF")
file = st.file_uploader("Please upload a PDF file", type=["pdf"])

if file is not None:
    st.write("Processing PDF...")
    images = pdf_to_images(file)
    
    for i, image in enumerate(images):
        st.image(image, caption=f"Page {i + 1}", use_column_width=True)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Perform OCR on each preprocessed page image
        ocr_result = perform_ocr(preprocessed_image)
        st.write(f"OCR Result for Page {i + 1}:")
        st.text(ocr_result)
