import os
import zipfile
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

from densenet_model import DenseNet121Model
from efficientnet_model import EfficientNetB0Model

# ✅ Step 1: Unzip ensemble_model.zip if .pth is not yet extracted
if not os.path.exists("ensemble_model.pth"):
    with zipfile.ZipFile("ensemble_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# ✅ Step 2: Define your ensemble model
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
        combined = torch.cat((x1, x2), dim=1)
        return self.classifier(combined)

# ✅ Step 3: Load model weights
model = EnsembleModel()
model.load_state_dict(torch.load("ensemble_model.pth", map_location="cpu"))
model.eval()

# ✅ Step 4: Define transformation and prediction function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

labels = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()
    return labels[pred]

# ✅ Step 5: Launch Gradio interface
iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="label", title="DR Classification")
iface.launch()
