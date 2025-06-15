import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetB0Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        self.base._fc = nn.Identity()

    def forward(self, x):
        return self.base(x)
