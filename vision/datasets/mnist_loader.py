import os
from torchvision.datasets import MNIST

def prepare_mnist_images(output_dir="mnist_images", count=100):
    os.makedirs(output_dir, exist_ok=True)
    mnist_test = MNIST(root="./data", train=False, download=True)
    image_paths = []

    for i, (img, label) in enumerate(mnist_test):
        if i >= count:
            break
        img = img.convert("RGB")  # Convert grayscale to RGB
        img_path = f"{output_dir}/{label}_{i}.png"
        img.save(img_path)
        image_paths.append(img_path)

    print(f"✅ {count} MNIST images saved to '{output_dir}'")
    return image_paths  # Return list of paths
