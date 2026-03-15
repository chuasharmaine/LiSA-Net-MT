import torch.nn as nn
from torchvision import models

def DenseNet121(num_classes, pretrained=True):
    model = models.densenet121(pretrained=pretrained)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model