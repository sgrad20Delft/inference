#
# from PIL import Image
# from torchvision import transforms
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to 224x224
#     # transforms.Grayscale(num_output_channels=3), # Convert 1-channel to 3-channel
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
#
# def custom_preprocess(image_path):
#     # print(f"[CUSTOM PREPROCESS] Handling image: {image_path}")
#     image = Image.open(image_path).convert("RGB")
#     tensor = transform(image).unsqueeze(0)
#     # print("[CUSTOM PREPROCESS] shape:", tensor.shape)
#     return tensor
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def preprocess(image_path, layout="NCHW"):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 416))
    image = image.astype(np.float32) / 255.0
    print(f"Image shape: {image.shape}")
    print(f"Layout: {layout}")
    if layout == "NCHW":
        image = np.transpose(image, (2, 0, 1))  # CHW
    # else keep as HWC (NHWC)

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return torch.from_numpy(image).float()


def preprocess_all(dataset_dir, output_dir=None):
    dataset_path = Path(dataset_dir)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Supported extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    all_images = sorted([f for f in dataset_path.iterdir() if f.suffix.lower() in image_extensions])
    print(f"[INFO] Found {len(all_images)} images in {dataset_path}")

    for img_path in tqdm(all_images, desc="Preprocessing"):
        try:
            tensor = preprocess(str(img_path))

            if output_dir:
                torch.save(tensor, output_path / (img_path.stem + ".pt"))

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {e}")

    print("Finished preprocessing all images.")
