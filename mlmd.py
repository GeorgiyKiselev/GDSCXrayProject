import os
from PIL import Image, ImageEnhance
from io import BytesIO

import pathlib
temp = pathlib.PosixPath

if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

import numpy as np
import pandas as pd
import pandas as pd
import requests
import streamlit as st
import torch
import torchvision.transforms as transforms

def get_x(r): 
    return pathlib.Path('images')/r['Image Index']

def get_y(r): 
    return r['Finding Labels'].split('|')

def load_diagnosis(path):
    df = pd.read_csv(path)
    return df['Diagnosis'].tolist()

def load_describ(path):
    df = pd.read_csv(path)
    diagnosis = df['Diagnosis'].tolist()
    information = df['Information'].tolist()
    
    diagnosis_information = {}
    for diag, info in zip(diagnosis, information):
        diagnosis_information[diag] = info
    
    return diagnosis_information

def get_prediction(model, image, class_names, class_info):
    transform = transforms.ToTensor()

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to('cpu')

    with torch.no_grad():
        outputs = model.model(image)
        _, preds = torch.max(outputs, 1)
        _, preds2 = torch.topk(outputs, 3, 1)
        prob = torch.softmax(outputs, 1)[0]

    if preds.item() == 10:
        pred_class = class_names[preds2[0][0].item()]
    else:
        pred_class = class_names[preds.item()]

    pred_info = class_info.get(pred_class, "No information available")
    return pred_class, pred_info, prob, preds
 
model = torch.load('xrmodel.pkl', map_location=torch.device('cpu'))
model.eval()


st.set_page_config(
    page_title="ml.MD",
    page_icon="💔",
    layout="centered"
)


st.title("💔 ml.MD 💔")
upload_tab, url_tab = st.tabs(["Upload", "Image URL"])

img = Image.open("./assets/defaultxrayHERNIA.jpeg")

with upload_tab:
    file = st.file_uploader("Upload Art", key="file_uploader")
    if file is not None:
        try:
            img = Image.open(file)
        except:
            st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the file uploader, remove the image URL first.")

with url_tab:
    url_text = st.empty()
    url = url_text.text_input("Image URL", key="image_url")
    if url!="":
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")

with img and st.expander("X-Ray", expanded=True):
    st.image(img, use_column_width=True)

if st.button("Analyze"):
    class_names = load_diagnosis('mdmlCSV.csv')
    class_info = load_describ('mdmlCSV.csv')

    if img.mode != 'RGB':
        img = img.convert('RGB')

    pred_class, pred_info, prob, preds = get_prediction(model, img, class_names, class_info)
    st.subheader("🩺Diagnosis: ")
    st.write(pred_class.upper())
    st.write("_{:.2f}% chance_".format(prob[preds[0].item()].item() * 100))
    st.subheader("📖Information: ")
    st.write(pred_info)
