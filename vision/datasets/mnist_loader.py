import os
from torchvision.datasets import MNIST

def prepare_mnist_images(output_dir="mnist_images", count=100):
    os.makedirs(output_dir, exist_ok=True)
    mnist_test = MNIST(root="./data", train=False, download=True)

    for i, (img, label) in enumerate(mnist_test):
        if i >= count:
            break
        # Just convert grayscale to RGB (no ToPILImage needed)
        img = img.convert("RGB")
        img.save(f"{output_dir}/{label}_{i}.png")

    print(f"✅ {count} MNIST images saved to '{output_dir}'")
    return output_dir
