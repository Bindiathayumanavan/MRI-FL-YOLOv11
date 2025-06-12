import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the model
model_path = 'tumor_classifier.pth'
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classifier
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Remove the final classifier layer to extract features
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data_dir = 'BrainMRI/val'
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

features = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        feats = feature_extractor(imgs).squeeze()
        features.append(feats)
        labels.extend(lbls.tolist())

features = torch
