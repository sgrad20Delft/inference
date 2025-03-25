from torchvision import transforms
from PIL import Image

def preprocess(image_path, transform=None):
    """Apply a user-defined or default transformation to an image."""
    image = Image.open(image_path).convert("RGB")
    
    # Use provided transform; fallback to a default one if None
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    return transform(image).unsqueeze(0)
