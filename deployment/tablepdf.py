import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytesseract
import fitz  # PyMuPDF
from matplotlib import pyplot as plt
from app2 import process_image_app

# Define transformations
TRANSFORM = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255,
    ),
    ToTensorV2()
])

# Define DenseNet architecture
class DenseNet(nn.Module):
    def __init__(self, weights="imagenet", requires_grad=True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(weights=weights).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):  
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8, 10):  
            self.densenet_out_2.add_module(str(x), denseNet[x])

        self.densenet_out_3.add_module(str(10), denseNet[10])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out_1 = self.densenet_out_1(x)  
        out_2 = self.densenet_out_2(out_1)  
        out_3 = self.densenet_out_3(out_2)  
        return out_1, out_2, out_3

# Define TableDecoder architecture
class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernels[0],
            stride=strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=kernels[1],
            stride=strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[2],
            stride=strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=1,
            kernel_size=kernels[3],
            stride=strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  
        out = self.upsample_1_table(x)  
        out = torch.cat((out, pool_4_out), dim=1)  
        out = self.upsample_2_table(out)  
        out = torch.cat((out, pool_3_out), dim=1)  
        out = self.upsample_3_table(out)  
        return out

# Define ColumnDecoder architecture
class ColumnDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(ColumnDecoder, self).__init__()
        self.conv_8_column = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernels[0], stride=strides[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernels[0], stride=strides[0])
        )
        self.upsample_1_column = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=kernels[1],
            stride=strides[1])
        self.upsample_2_column = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[2],
            stride=strides[2])
        self.upsample_3_column = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=1,
            kernel_size=kernels[3],
            stride=strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_8_column(x)  
        out = self.upsample_1_column(x)  
        out = torch.cat((out, pool_4_out), dim=1)  
        out = self.upsample_2_column(out)  
        out = torch.cat((out, pool_3_out), dim=1)  
        out = self.upsample_3_column(out)  
        return out

# Define the main network
class TableNet(nn.Module):
    def __init__(self):
        super(TableNet, self).__init__()

        self.base_model = DenseNet(weights=None, requires_grad=True)
        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1, 1), (1, 1), (2, 2), (16, 16)]
        self.strides = [(1, 1), (1, 1), (2, 2), (16, 16)]

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self.table_decoder = TableDecoder(self.pool_channels, self.kernels, self.strides)
        self.column_decoder = ColumnDecoder(self.pool_channels, self.kernels, self.strides)

    def forward(self, x):
        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out)  
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out)  
        column_out = self.column_decoder(conv_out, pool_3_out, pool_4_out)  
        return table_out, column_out

# Load model
@st.cache_resource()
def load_model():
    model = TableNet()
    model.load_state_dict(torch.load("densenet_config_4_model_checkpoint.pth.tar", map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    return model

# Perform OCR
def perform_ocr(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    config = '--psm 6'
    return pytesseract.image_to_string(image, lang='eng', config=config)

# Convert PDF to images
def pdf_to_images(pdf_file):
    pdf_data = pdf_file.read()
    doc = fitz.open("pdf", pdf_data)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# Column Image Processing
def process_column_image(column_crop):
    # Convert to grayscale
    column_image = np.array(column_crop.convert('L'))
    
    # Thresholding the image to a binary image
    _, column_bin = cv2.threshold(column_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Inverting the image
    column_bin = 255 - column_bin
    
    return column_bin

def predict(image):
    with st.spinner('Processing...'):
        orig_image = image.resize((1024, 1024))

        # Ensure image is in RGB format
        if orig_image.mode != 'RGB':
            orig_image = orig_image.convert('RGB')

        test_img = np.array(orig_image.convert('LA').convert("RGB"))

        now = datetime.now()
        image = TRANSFORM(image=test_img)["image"]
        with torch.no_grad():
            image = image.unsqueeze(0)
            model = load_model()
            table_out, column_out = model(image)
            table_out = torch.sigmoid(table_out)
            column_out = torch.sigmoid(column_out)

        table_out = (table_out.detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(np.uint8)
        column_out = (column_out.detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(np.uint8)

        table_contours, _ = cv2.findContours(table_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        column_contours, _ = cv2.findContours(column_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not column_contours:
            st.write("No column contours found. Skipping page.")
            return

        # Process each column
        image_with_contours = np.array(orig_image)
        color_table = (0, 255, 0)
        color_column = (255, 0, 0)

        for contour in table_contours:
            cv2.drawContours(image_with_contours, [contour], -1, color_table, 2)

        for contour in column_contours:
            cv2.drawContours(image_with_contours, [contour], -1, color_column, 2)

        st.image(image_with_contours, caption="Image with Table and Column Contours", use_column_width=True)

        # Find the bounding box of the table
        table_x, table_y, table_w, table_h = cv2.boundingRect(table_contours[0])

        column_regions = []
        sorted_column_regions = []

        df = pd.DataFrame()
        biglist = []

        # Sort column contours by their bounding rectangle's x-coordinate
        for i, contour in enumerate(sorted(column_contours, key=lambda c: cv2.boundingRect(c)[0])):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Adjust column bounding box to match the table's top and bottom
            column_region = column_out[table_y:table_y + table_h, x:x + w]

            # Check for content and minimum area threshold
            if not np.any(column_region) or w * h < 100:  # Adjust the area threshold as needed
                st.write(f"## Column {i + 1} (Empty or too small, skipping)")
                continue

            # Use OpenCV for cropping to maintain quality
            orig_image_np = np.array(orig_image)
            
            try:
                # Use OpenCV for cropping to maintain quality
                column_crop = orig_image_np[y:y + h, x:x + w]
                column_crop_pil = Image.fromarray(column_crop)

                # Detect horizontal lines in the column crop
                img_bin = cv2.cvtColor(column_crop, cv2.COLOR_RGB2GRAY)
                hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
                horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
                cv2.imwrite(f"column_{i + 1}_horizontal.jpg", horizontal_lines)
                
                # Column Image Processing
                list1 = process_image_app(column_crop)
                st.image(column_crop_pil, caption=f"Processed Column {i + 1}", use_column_width=True)
                print("List size is ", len(list1), "\n")
                biglist.append(list1)

            except Exception as e:
                st.write(f"## Error processing Column {i + 1}: not a proper table")
                continue

        # Convert 2D array to DataFrame
        df = pd.DataFrame(biglist)

        # Transpose and set first row as header
        df = df.transpose()
        new_header = df.iloc[0]  # Grab the first row for the header
        df = df[1:]  # Take the data less the header row
        df.columns = new_header  # Set the header row as the dataframe header

        # Streamlit app setup
        st.header("Display DataFrame in Streamlit")

        # Display the DataFrame in Streamlit
        st.write(df)

        end_time = datetime.now()
        difference = end_time - now
        time = "{}".format(difference)
        st.write(f"Processing time: {time} secs")

# Streamlit app setup.
st.header("Data Extraction from Tables")

file = st.file_uploader("Please upload a PDF file", type=["pdf"])

if file is not None:
    images = pdf_to_images(file)
    for i, image in enumerate(images):
        st.write(f"Processing page {i + 1}/{len(images)}")
        predict(image)
