
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import zipfile
import os

from densenet_model import DenseNet121Model
from efficientnet_model import EfficientNetB0Model

# Step 1: Extract the model
if not os.path.exists("ensemble_model.pth"):
    with zipfile.ZipFile("ensemble_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Step 2: Define model
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.densenet = DenseNet121Model()
        self.efficientnet = EfficientNetB0Model()
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.efficientnet(x)
        return self.classifier(torch.cat((x1, x2), dim=1))

# Step 3: Load model
model = EnsembleModel()
model.load_state_dict(torch.load("ensemble_model.pth", map_location="cpu"))
model.eval()

# Step 4: Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
labels = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# Step 5: Streamlit UI
st.title("Diabetic Retinopathy Classification")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = labels[output.argmax(1).item()]

    st.success(f"Predicted Class: **{prediction}**")
