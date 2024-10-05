import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    """Helper function to display images using matplotlib"""
    plt.figure(figsize=(10, 6))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:  # BGR image, convert it to RGB
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def count_remaining_contours(dilate):
    """Count contours in a dilated image"""
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return len(cnts), cnts

# Step 1: Load image and make a copy
image_path = '6.png'
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Image not found or cannot be opened: {image_path}")

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

# Display results of both filtering methods
# Clone original image to draw bounding boxes separately for comparison
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

# Display the images with bounding boxes for both methods
display_image('Mean-Based Filtering Result', image_mean)
display_image('Median-Based Filtering Result', image_median)
print(count_mean, count_median)

# Display the final result with the lowest number of regions, ignoring 1 region (whole)
if (count_mean > 1 and count_median == 1) or (count_mean > 1 and count_mean < count_median):
    print(f"Mean-Based Filtering Selected with {count_mean} contours")
    display_image('Final Result - Mean-Based Filtering', image_mean)
elif (count_median > 1 and count_mean == 1) or (count_median > 1 and count_median < count_mean):
    print(f"Median-Based Filtering Selected with {count_median} contours")
    display_image('Final Result - Median-Based Filtering', image_median)
else:
    print("No contours found in either filtering approach.")

