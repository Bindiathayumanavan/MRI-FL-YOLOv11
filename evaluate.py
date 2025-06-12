import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.classifier import build_classifier_model
import os

# Path to validation dataset and model file
val_dir = 'BrainMRI/val'
model_path = 'tumor_classifier.pth'

# Image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load validation dataset
val_data = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_classifier_model(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
