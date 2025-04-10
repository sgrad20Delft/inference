from pathlib import Path

from datasets import load_dataset
from PIL import Image
import os, json

# Step 1: Load ImageNette
dataset = load_dataset("frgfm/imagenette", "160px", split="train")

# Step 2: Map ImageNette WNIDs to ImageNet-1k class indices
imagenette_to_imagenet_idx = {
  "n01440764": 0,
  "n02102040": 217,
  "n02979186": 482,
  "n03000684": 491,
  "n03028079": 497,
  "n03394916": 566,
  "n03417042": 569,
  "n03425413": 571,
  "n03445777": 574,
  "n03888257": 701

}

# Fixed mapping: label index to WNID (based on ImageNette class order)
label_id_to_wnid = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"
]

# Step 3: Save to disk
base = Path(__file__).resolve().parent
base_dir=base.parents[3]
print(base_dir)
output_dir = os.path.join(base_dir, "dataset_dir", "imagenette","imagenette_images")
label_dir = os.path.join(base_dir, "dataset_dir", "imagenette","labels")
os.makedirs(label_dir, exist_ok=True)

label_map = {}

for i, sample in enumerate(dataset):
    print(f"sample {sample}")
    img = sample["image"]
    label_id = sample["label"]
    wnid = label_id_to_wnid[label_id]              # ✅ Safe mapping
    imagenet_idx = imagenette_to_imagenet_idx[wnid]
    # print(f"[DEBUG] image: img_{i}.jpg | label_id: {label_id} | WNID: {wnid} | ImageNet-1k index: {imagenet_idx}")
    assert wnid in imagenette_to_imagenet_idx, f"WNID {wnid} not in mapping!"
    filename = f"img_{i}.jpg"
    filepath = os.path.join(output_dir, filename)

    img.save(filepath)

    label_map[filename] = imagenet_idx

unique_labels_used = set(label_map.values())
print(f"[INFO] Unique ImageNet class indices used: {sorted(unique_labels_used)}")

with open(os.path.join(label_dir, "labels_imagenette.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# Step 4: Save labels.json
def proces_custom_labels():
    labels_dict = {}
    for filename, label_idx in label_map.items():
        labels_dict[filename] = label_idx
    return labels_dict



print("Imagenette Dataset saved under:", output_dir)
