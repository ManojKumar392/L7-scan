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
import fitz  # PyMuPDF for handling PDFs
from linedetection import process_image_app  # Import line detection function
from pandas.io.parsers.base_parser import ParserBase

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

# Define TableDecoder and ColumnDecoder architectures
class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(256, 256, kernels[0], strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(256, 128, kernels[1], strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(128 + channels[0], 256, kernels[2], strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(256 + channels[1], 1, kernels[3], strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  
        out = self.upsample_1_table(x)  
        out = torch.cat((out, pool_4_out), dim=1)  
        out = self.upsample_2_table(out)  
        out = torch.cat((out, pool_3_out), dim=1)  
        out = self.upsample_3_table(out)  
        return out

class ColumnDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(ColumnDecoder, self).__init__()
        self.conv_8_column = nn.Sequential(
            nn.Conv2d(256, 256, kernels[0], strides[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(256, 256, kernels[0], strides[0])
        )
        self.upsample_1_column = nn.ConvTranspose2d(256, 128, kernels[1], strides[1])
        self.upsample_2_column = nn.ConvTranspose2d(128 + channels[0], 256, kernels[2], strides[2])
        self.upsample_3_column = nn.ConvTranspose2d(256 + channels[1], 1, kernels[3], strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_8_column(x)  
        out = self.upsample_1_column(x)  
        out = torch.cat((out, pool_4_out), dim=1)  
        out = self.upsample_2_column(out)  
        out = torch.cat((out, pool_3_out), dim=1)  
        out = self.upsample_3_column(out)  
        return out

# Define main network
class TableNet(nn.Module):
    def __init__(self):
        super(TableNet, self).__init__()
        self.base_model = DenseNet(weights=None, requires_grad=True)
        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1, 1), (1, 1), (2, 2), (16, 16)]
        self.strides = [(1, 1), (1, 1), (2, 2), (16, 16)]

        self.conv6 = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(256, 256, (1, 1)),
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

def deduplicate_columns(df):
    """
    Deduplicate column names by appending suffixes to duplicate columns.
    """
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    return df

# Load the model
@st.cache_resource()
def load_model():
    model = TableNet()
    model.load_state_dict(torch.load("densenet_config_4_model_checkpoint.pth.tar", map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    return model

MIN_WIDTH = 50  # Adjust as needed
MIN_HEIGHT = 50  # Adjust as needed

# Prediction function
def predict(image):
    st.write("Processing Image...")

    orig_image = image.resize((1024, 1024))
    if orig_image.mode != 'RGB':
        orig_image = orig_image.convert('RGB')

    test_img = np.array(orig_image.convert('LA').convert("RGB"))
    transformed_image = TRANSFORM(image=test_img)["image"]
    
    with torch.no_grad():
        transformed_image = transformed_image.unsqueeze(0)
        model = load_model()
        table_out, column_out = model(transformed_image)
        table_out = torch.sigmoid(table_out)
        column_out = torch.sigmoid(column_out)

    table_out = (table_out.detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(np.uint8)
    column_out = (column_out.detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(np.uint8)

    table_contours, _ = cv2.findContours(table_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    column_contours, _ = cv2.findContours(column_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not table_contours or not column_contours:
        st.write("No table or column contours found. Skipping page.")
        return

    image_with_contours = np.array(orig_image)
    color_table = (0, 255, 0)
    color_column = (255, 0, 0)

    for contour in table_contours:
        cv2.drawContours(image_with_contours, [contour], -1, color_table, 2)

    for contour in column_contours:
        cv2.drawContours(image_with_contours, [contour], -1, color_column, 2)

    st.image(image_with_contours, caption="Image with Table and Column Contours", use_column_width=True)

    biglist = []
    for i, contour in enumerate(sorted(column_contours, key=lambda c: cv2.boundingRect(c)[0])):
        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue
        
        # Crop from the original unaltered image
        column_crop = orig_image.crop((x, y, x + w, y + h))  # Use PIL Image.crop method for clean cropping
        
        # Pass the cropped region to the OCR function
        list1 = process_image_app(column_crop)
        
        # For visualization, draw the bounding box on the image with contours
        cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (36, 255, 12), 3)
        
        # Display the cropped image for inspection or debugging
        st.image(column_crop, caption=f"Processed Column {i + 1}", use_column_width=True)
        
        # Append the OCR results to the biglist
        biglist.append(list1)


    if biglist:
        df = pd.DataFrame(biglist).transpose()
        df.columns = df.iloc[0].fillna("Unnamed")
        df = df[1:]
        st.write("Extracted Table Data")
        df = deduplicate_columns(df)
        st.write(df)

# Streamlit app
def main():
    st.title("Table Detection and OCR")

    uploaded_file = st.file_uploader("Choose an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            pdf_data = fitz.open("pdf", uploaded_file.read())
            for page_num in range(len(pdf_data)):
                page = pdf_data.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                predict(img)
        else:
            image = Image.open(uploaded_file)
            predict(image)

if __name__ == "__main__":
    main()
