import torch.nn as nn
from torchvision import models

def ResNet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)

    # replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model