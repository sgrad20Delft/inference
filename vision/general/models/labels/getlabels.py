import os
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# Paths to your MNIST filesimport os
base = Path(__file__).resolve().parent
base_dir=base.parents[2]
print(base_dir)# Gets the directory of your script
image_file = os.path.join(base_dir, "dataset_dir", "mnist", "train-images-idx3-ubyte", "train-images-idx3-ubyte")
label_file = os.path.join(base_dir, "dataset_dir", "mnist", "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")
output_dir = os.path.join(base_dir, "dataset_dir", "mnist","mnist_images")
label_dir = os.path.join(base_dir, "dataset_dir", "mnist","labels")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
# Load images
with open(image_file, "rb") as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

# Load labels
with open(label_file, "rb") as f:
    magic, num = struct.unpack(">II", f.read(8))
    label_data = np.frombuffer(f.read(), dtype=np.uint8)

# Save images and build label dict
labels_dict = {}
for i in range(len(image_data)):
    label = label_data[i]
    filename = f"{label}_{i:05d}.png"
    filepath = os.path.join(output_dir, filename)
    Image.fromarray(image_data[i]).save(filepath)
    labels_dict[filename] = int(label)

# Save labels dict
with open(os.path.join(label_dir,"labels.json"), "w") as f:
    json.dump(labels_dict, f, indent=2)

print(f"Saved {len(labels_dict)} images and labels to '{output_dir}'")
