import cv2
import pytesseract
import numpy as np

def process_image_app(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to highlight lines
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Determine dynamic kernel sizes based on image dimensions
    height, width = img.shape[:2]
    horizontal_kernel_size = (width // 30, 1)
    vertical_kernel_size = (1, height // 30)
    
    # Apply morphological operations to enhance horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, horizontal_kernel_size)
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Apply morphological operations to enhance vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, vertical_kernel_size)
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine horizontal and vertical lines
    combined_lines = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0.0)

    # Dilate the combined lines to ensure they are continuous
    dilated_lines = cv2.dilate(combined_lines, np.ones((3, 3), np.uint8), iterations=1)

    # Use Canny edge detection
    edges = cv2.Canny(dilated_lines, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # Separate lines into horizontal and vertical
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line
                horizontal_lines.append((x1, y1, x2, y2))

    # Sort horizontal lines by their coordinates
    horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1])

    # Define the vertical lines as the left and right edges of the image
    image_height, image_width = img.shape[:2]
    vertical_lines = [(0, 0, 0, image_height), (image_width, 0, image_width, image_height)]

    # Initialize an empty list to store all cell texts
    all_cell_texts = []

    if not horizontal_lines:
        return all_cell_texts  # Return empty list if no horizontal lines are found

    try:
        # Extract rows and columns from detected grid lines
        for i in range(len(horizontal_lines) - 1):
            y1, y2 = horizontal_lines[i][1], horizontal_lines[i + 1][1]
            x1, x2 = vertical_lines[0][0], vertical_lines[1][0]

            # Ensure we do not extract regions outside the image boundaries
            if y2 > y1 and x2 > x1:
                cell_img = img[y1:y2, x1:x2]

                # Preprocess the cell image for better OCR results
                column_image = np.array(cell_img)
                column_image = cv2.cvtColor(column_image, cv2.COLOR_BGR2GRAY)
                _, column_bin = cv2.threshold(column_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                column_bin = 255 - column_bin

                # Perform OCR on the preprocessed cell image
                cell_text = pytesseract.image_to_string(column_bin, config='--psm 6').strip()

                # Add non-empty cell text to the list
                if cell_text:
                    all_cell_texts.append(cell_text)

    except Exception as e:
        print(f"Error processing image: {e}")
        return []  # Return empty list on error

    # Optionally, return OCR results or any other processed data
    return all_cell_texts

# Example usage:
# Uncomment and modify as needed
# img = cv2.imread("C:\\Users\\manoj\\OneDrive\\Pictures\\Pictures\\Screenshots\\Column.png")
# processed_data = process_image_app(img)
# for idx, cell_text in enumerate(processed_data, start=1):
#     print(f"Row {idx}: {cell_text}")
