import streamlit as st
from PIL import Image
import torch
import pandas as pd
import torchvision.transforms as transforms

def get_x(r): 
    return Path('images')/r['Image Index']

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
        prob = torch.softmax(outputs, 1)[0]

    
    pred_class = class_names[preds.item()]
    pred_info = class_info.get(pred_class, "No information available")
    return pred_class, pred_info, prob

def main():
    model = torch.load('xrmodel.pkl', map_location=torch.device('cpu'))
    model.eval()

    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff;
        }

        .title {
            color: #264653;
            text-align: center;
            font-size: 36px;
            margin-bottom: 30px;
        }

        .upload-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 50px;
            margin-bottom: 50px;
        }

        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }

        .prediction-container {
            text-align: center;
            margin-top: 50px;
            margin-bottom: 50px;
        }

        .predicted-class {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        .information {
            font-size: 18px;
            color: #2a9d8f;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="title">🌟 X-ray Chest Scan Diagnosis 🌟</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        class_names = load_diagnosis('mdmlCSV.csv')
        class_info = load_describ('mdmlCSV.csv')

        pred_class, pred_info, pred_prob = get_prediction(model, image, class_names, class_info)
        st.write("Probabiloities: ", pred_prob)
        st.write("🩺Diagnosis: ", pred_class)
        st.write("📖Information: ", pred_info)

if __name__ == "__main__":
    main()
