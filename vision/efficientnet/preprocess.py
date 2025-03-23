from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)
