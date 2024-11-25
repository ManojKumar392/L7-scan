import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

# Configure the Gemini API key
api_key = "key"
genai.configure(api_key=api_key)

# Function to extract text using Gemini API
# Function to extract text using Gemini API with proper segmentation
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
            "Each distinct text component (e.g., axis labels, legends, or individual values) "
            "should appear on a new line. Do not include any additional information or explanations."
        )
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, pil_image])
        extracted_text = response.text.strip()
        return extracted_text
    except Exception as e:
        return f"Error in Gemini API: {e}"

# Streamlit UI
st.title("Graph Data Detection and Extraction Tool")
uploaded_file = st.file_uploader("Upload an image of a graph", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Step 1: Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    original = image.copy()

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Otsu's thresholding to binarize the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Step 4: Dilate the image with a horizontal kernel to connect words/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Step 5: Filter small contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    min_area = 2000
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

    def count_remaining_contours(dilate):
        """Count contours in a dilated image"""
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return len(cnts), cnts


    # Function for compactness-based filtering
    def compactness_filter(dilate, method="median"):
        """
        Apply compactness-based filtering using areas instead of compactness ratios.
        """
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Collect areas of contours
        areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) >= min_area]

        if areas:
            if method == "mean":
                mean_area = np.mean(areas)
                std_area = np.std(areas)
                compactness_threshold = mean_area / (mean_area + std_area) if (mean_area + std_area) != 0 else 0.65
            else:  # Default to median
                compactness_threshold = np.median(areas) + 0.0001
        else:
            compactness_threshold = 0.65

        # Filter contours based on calculated compactness threshold
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            bounding_box_area = w * h

            if bounding_box_area > 0:  # To avoid division by zero
                compactness = area / bounding_box_area
                if compactness > compactness_threshold:
                    cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

        return dilate

    # Apply filtering methods
    dilate_mean = compactness_filter(dilate.copy(), method="mean")
    dilate_median = compactness_filter(dilate.copy(), method="median")

    # Count contours
    count_mean, cnts_mean = count_remaining_contours(dilate_mean)
    count_median, cnts_median = count_remaining_contours(dilate_median)

    # Draw bounding boxes for results
    image_mean = original.copy()
    image_median = original.copy()

    for c in cnts_mean:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.95 * image.shape[1] and h < 0.95 * image.shape[0]:
            cv2.rectangle(image_mean, (x, y), (x + w, y + h), (36, 255, 12), 2)

    for c in cnts_median:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.95 * image.shape[1] and h < 0.95 * image.shape[0]:
            cv2.rectangle(image_median, (x, y), (x + w, y + h), (36, 255, 12), 2)

    # Final decision logic
    # Final decision logic with OCR integration
    print(count_mean)
    print(count_median)
    if count_mean <= count_median:
        st.write(f"Mean-Based Filtering Selected with {count_mean} contours")
        st.write("Detected Regions with Extracted Text:")
        for c in cnts_mean:
            x, y, w, h = cv2.boundingRect(c)
            if w < 0.95 * image.shape[1] and h < 0.95 * image.shape[0]:
                # Draw the bounding box
                cv2.rectangle(image_mean, (x, y), (x + w, y + h), (36, 255, 12), 2)
                # Extract text using OCR
                extracted_text = extract_text_with_gemini(image, roi=(x, y, w, h))
                st.write(f"Region at (x={x}, y={y}, w={w}, h={h}): {extracted_text}")
        st.image(cv2.cvtColor(image_mean, cv2.COLOR_BGR2RGB), caption="Mean-Based Filtering Result")
    else:
        st.write(f"Median-Based Filtering Selected with {count_median} contours")
        st.write("Detected Regions with Extracted Text:")
        for c in cnts_median:
            x, y, w, h = cv2.boundingRect(c)
            if w < 0.95 * image.shape[1] and h < 0.95 * image.shape[0]:
                # Draw the bounding box
                cv2.rectangle(image_median, (x, y), (x + w, y + h), (36, 255, 12), 2)
                # Extract text using OCR
                extracted_text = extract_text_with_gemini(image, roi=(x, y, w, h))
                st.write(f"Region at (x={x}, y={y}, w={w}, h={h}): {extracted_text}")
        st.image(cv2.cvtColor(image_median, cv2.COLOR_BGR2RGB), caption="Median-Based Filtering Result")


