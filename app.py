import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

from densenet_model import DenseNet121Model
from efficientnet_model import EfficientNetB0Model

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

model = EnsembleModel()
model.load_state_dict(torch.load("ensemble_model.pth", map_location="cpu"))
model.eval()

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

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="label", title="DR Classification")
iface.launch()
