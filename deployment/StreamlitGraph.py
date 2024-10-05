import streamlit as st
import cv2
import numpy as np

# Function to display images using streamlit
def display_image_st(title, image):
    """Helper function to display images using streamlit"""
    st.write(title)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def count_remaining_contours(dilate):
    """Count contours in a dilated image"""
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return len(cnts), cnts

# Streamlit UI
st.title('Graph/Image Detection')
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Step 1: Load image from uploaded file
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

    # Step 5: Find contours for size-based filtering
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    min_area = 2000  # Set minimum area to filter out small contours

    # Filter small contours
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

    # Function to apply mean-based compactness filtering
    def mean_based_compactness_filter(dilate):
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) >= min_area]

        # Calculate mean and std of areas
        if areas:
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            compactness_threshold = mean_area / (mean_area + std_area) if (mean_area + std_area) != 0 else 0.65
        else:
            compactness_threshold = 0.65
        
        # Filter contours based on compactness
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            bounding_box_area = w * h

            if bounding_box_area > 0:  # To avoid division by zero
                compactness = area / bounding_box_area
                if compactness > compactness_threshold:
                    cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

        return dilate

    # Function to apply median-based compactness filtering
    def median_based_compactness_filter(dilate):
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Calculate compactness for each contour
        compactness_ratios = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            bounding_box_area = w * h
            if bounding_box_area > 0:
                compactness = area / bounding_box_area
                compactness_ratios.append(compactness)

        if compactness_ratios:
            median_compactness = np.median(compactness_ratios)
            adjustment_factor = 0.0001  # Fine-tune as needed
            compactness_threshold = median_compactness + adjustment_factor
        else:
            compactness_threshold = 0.65

        # Filter contours based on improved compactness threshold
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            bounding_box_area = w * h

            if bounding_box_area > 0:
                compactness = area / bounding_box_area
                if compactness > compactness_threshold:
                    cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

        return dilate

    # Clone the dilated image for each approach
    dilate_mean = dilate.copy()
    dilate_median = dilate.copy()

    # Apply both filtering approaches
    dilate_mean = mean_based_compactness_filter(dilate_mean)
    dilate_median = median_based_compactness_filter(dilate_median)

    # Count remaining contours in each approach
    count_mean, cnts_mean = count_remaining_contours(dilate_mean)
    count_median, cnts_median = count_remaining_contours(dilate_median)

    # Clone original image to draw bounding boxes for final output
    image_mean = original.copy()
    image_median = original.copy()

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Draw bounding boxes for mean-based filtering result
    for i, c in enumerate(cnts_mean):
        x, y, w, h = cv2.boundingRect(c)
        
        # Skip bounding boxes that are almost as large as the whole image
        if w >= 0.95 * image_width and h >= 0.95 * image_height:
            continue  # Skip this bounding box
        
        cv2.rectangle(image_mean, (x, y), (x + w, y + h), (36, 255, 12), 3)

    # Draw bounding boxes for median-based filtering result
    for i, c in enumerate(cnts_median):
        x, y, w, h = cv2.boundingRect(c)
        
        # Skip bounding boxes that are almost as large as the whole image
        if w >= 0.95 * image_width and h >= 0.95 * image_height:
            continue  # Skip this bounding box
        
        cv2.rectangle(image_median, (x, y), (x + w, y + h), (36, 255, 12), 3)

    # Display the final result with the lowest number of regions, ignoring 1 region (whole)
    if (count_mean > 1 and count_median == 1) or (count_mean > 1 and count_mean < count_median):
        st.write(f"Mean-Based Filtering Selected with {count_mean} contours")
        display_image_st('Final Result - Mean-Based Filtering', image_mean)
    elif (count_median > 1 and count_mean == 1) or (count_median > 1 and count_median < count_mean):
        st.write(f"Median-Based Filtering Selected with {count_median} contours")
        display_image_st('Final Result - Median-Based Filtering', image_median)
    else:
        st.write("No contours found in either filtering approach.")
