import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import google.generativeai as genai

# Configure the Gemini API key
api_key = "API-KEY"
genai.configure(api_key=api_key)

# Function to extract text using Gemini API
def extract_text_with_gemini(image, roi=None):
    """
    Extract text using Gemini API and ensure different parts of the text
    (e.g., axis labels, numbers, and legends) are separated by new lines.
    """
    if roi is not None:
        x, y, w, h = roi
        cropped = image[y:y+h, x:x+w]
    else:
        cropped = image

    # Convert the cropped image to PIL format
    pil_image = Image.fromarray(cropped)

    try:
        prompt = (
            "Extract the text content from the image and separate it logically. "
            "Each distinct text component (e.g., axis labels(both mandatorily), legends, or individual values) "
            "should appear on a new line. Do not include any additional information or explanations."
        )
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, pil_image])
        extracted_text = response.text.strip()
        return extracted_text
    except Exception as e:
        return f"Error in Gemini API: {e}"

# Function to find outliers using IQR
def find_outliers_with_iqr(cnts, min_area=2000, image_shape=None):
    """Find outliers based on compactness ratio using IQR."""
    compactness_ratios = []
    bounding_boxes = []

    # Calculate compactness ratios for each contour
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        bounding_box_area = w * h

        # Ignore regions covering almost the entire image
        if image_shape is not None:
            image_height, image_width = image_shape
            if w >= 0.95 * image_width and h >= 0.95 * image_height:
                continue  # Skip regions that are too large

        if area >= min_area and bounding_box_area > 0:
            compactness = area / bounding_box_area
            compactness_ratios.append(compactness)
            bounding_boxes.append((x, y, w, h))

    # Calculate IQR to find compactness outliers
    if compactness_ratios:
        q1 = np.percentile(compactness_ratios, 25)
        q3 = np.percentile(compactness_ratios, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out contours that are outliers
        outlier_indices = [
            i for i, ratio in enumerate(compactness_ratios)
            if ratio < lower_bound or ratio > upper_bound
        ]
        outlier_boxes = [bounding_boxes[i] for i in outlier_indices]
    else:
        outlier_boxes = []

    return outlier_boxes

# Function to process the image
def process_image(input_image):
    """Process the image and find graph-like regions using outlier detection."""
    # Step 1: Convert to OpenCV format
    image = np.array(input_image)
    original = image.copy()

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Otsu's thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Step 4: Dilate with a horizontal kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Step 5: Find contours in the dilated image
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Step 6: Detect outliers using compactness ratio
    outlier_boxes = find_outliers_with_iqr(cnts, min_area=2000, image_shape=image.shape[:2])

    # Draw bounding boxes around outliers and extract text using OCR
    image_with_outliers = original.copy()
    extracted_texts = []  # List to store extracted texts
    for x, y, w, h in outlier_boxes:
        cv2.rectangle(image_with_outliers, (x, y), (x + w, y + h), (0, 0, 255), 2)
        extracted_text = extract_text_with_gemini(original, roi=(x, y, w, h))
        extracted_texts.append((x, y, w, h, extracted_text))  # Store region and text

    return image_with_outliers, len(outlier_boxes), extracted_texts

# Function to convert PDF to images
def pdf_to_images(pdf_file):
    """Convert a PDF to a list of images."""
    pdf_data = pdf_file.read()  # Read the PDF file
    doc = fitz.open("pdf", pdf_data)  # Open PDF using PyMuPDF
    images = []

    # Iterate over each page in the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load a page
        pix = page.get_pixmap()  # Render page to a pixel map
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image
        images.append(img)

    return images

# Streamlit App
st.title("Graph Region Detection with OCR")
uploaded_file = st.file_uploader("Upload an image or PDF file", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        # Handle PDF input
        st.write("Uploaded file is a PDF.")
        images = pdf_to_images(uploaded_file)

        for page_num, image in enumerate(images):
            st.write(f"Processing Page {page_num + 1}...")
            processed_image, num_regions, extracted_texts = process_image(image)

            if num_regions > 0:
                st.write(f"Detected {num_regions} graph-like regions on page {page_num + 1}.")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=f"Page {page_num + 1} - Graph Regions")
                for region in extracted_texts:
                    x, y, w, h, text = region
                    st.write(f"Region (x={x}, y={y}, w={w}, h={h}): {text}")
            else:
                st.write(f"No graph-like regions detected on page {page_num + 1}.")
    else:
        # Handle image input
        st.write("Uploaded file is an image.")
        input_image = Image.open(uploaded_file)
        processed_image, num_regions, extracted_texts = process_image(input_image)

        if num_regions > 0:
            st.write(f"Detected {num_regions} graph-like regions.")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Graph Regions Detected")
            for region in extracted_texts:
                x, y, w, h, text = region
                st.write(f"Region (x={x}, y={y}, w={w}, h={h}): {text}")
        else:
            st.write("No graph-like regions detected.")
