import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from models.classifier import build_classifier_model
from utils.train_utils import train_model

# Directories
train_dir = 'BrainMRI/train'
val_dir = 'BrainMRI/val'
model_save_path = 'tumor_classifier.pth'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Build model
model = build_classifier_model(num_classes=2).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved as {model_save_path}")
