import torch.nn as nn
from torchvision import models

class DenseNet121Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Identity()

    def forward(self, x):
        return self.base(x)

