import torch.nn as nn
from torchvision import models

def build_classifier_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
