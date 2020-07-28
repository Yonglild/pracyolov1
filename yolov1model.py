import torch.nn as nn
from VGG import VGG

class YOLOV1(nn.Module):
    def __init__(self):
        super(YOLOV1, self).__init__()
        self.backbone = VGG.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, 7*7*5)

    def forward(self, x):
        x = self.backbone(x)    # (b, 512, 16, 16)
        x = self.avgpool(x)     # (b, 512, 7, 7)
        x = x.view(x.size(0), -1)   # (b, 4096)
        x = self.fc(x)
        x = x.view(x.size(0), 7, 7, 5)  # (b, 7, 7, 5)
        return x

