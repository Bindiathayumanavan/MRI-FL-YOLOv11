import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

def preprocess_image(image_path, to_tensor=True, resize=(224, 224), grayscale=False):
    """
    Preprocesses an image for inference or training.

    Args:
        image_path (str): Path to the image.
        to_tensor (bool): Whether to return a PyTorch tensor.
        resize (tuple): Size to resize the image to.
        grayscale (bool): Convert image to grayscale.

    Returns:
        torch.Tensor or np.ndarray: Preprocessed image.
    """
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.stack((image,) * 3, axis=-1)  # Convert back to 3 channels

    # Resize image
    image = cv2.resize(image, resize)

    if to_tensor:
        # Convert to tensor with normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        return transform(Image.fromarray(image))
    else:
        return image
