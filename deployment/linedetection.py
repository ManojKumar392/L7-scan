import cv2
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("The GOOGLE_API_KEY environment variable is not set.")
genai.configure(api_key=api_key)

def process_image_app(img, save_dir="..\\processed_images"):
    try:
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Save the entire column image for processing with Google Gemini
        column_img_path = os.path.join(save_dir, 'column.png')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(column_img_path, img_np)

        # Use Google Gemini for OCR
        try:
            column_pil_img = Image.open(column_img_path)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            prompt = """The cells in the image are separated by either spaces or visible lines. Please extract the content of the cells and provide the text in the following format:

Each cell's content should be on its own line.
The text from the first cell should appear on the first line, the text from the second cell on the second line, and so on.
Ignore any empty or blank cells."""
            response = model.generate_content([prompt, column_pil_img])
            print(response)
            csv_text = response.text.strip()
        except Exception as e:
            print(f"Error using Google Gemini API: {e}")
            return []

        # Save the CSV content
        csv_path = os.path.join(save_dir, 'column.csv')
        with open(csv_path, 'w') as csv_file:
            csv_file.write(csv_text)

        # Read the CSV content and return it as a list of rows
        all_cell_texts = []
        with open(csv_path, 'r') as csv_file:
            for line in csv_file:
                all_cell_texts.append(line.strip())

        return all_cell_texts

    except Exception as e:
        print(f"Error processing image: {e}")
        return []  # Return empty list on error

# Example usage:
# Uncomment and modify as needed
# processed_data = process_image_app(img_object, save_dir="path_to_save_directory")
# for idx, cell_text in enumerate(processed_data, start=1):
#     print(f"Row {idx}: {cell_text}")
