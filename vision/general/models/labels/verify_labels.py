import urllib.request
import json

# Step 1: Load ImageNet-1k class names
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# Step 2: Target classes
target_classes = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]

# Step 3: Lookup and append to a list
name_to_index = {name: idx for idx, name in enumerate(imagenet_classes)}
matched_indices = []

for cls in target_classes:
    idx = name_to_index.get(cls)
    if idx is not None:
        matched_indices.append(idx)
        print(f"{cls}: {idx}")
    else:
        print(f"{cls}: Not found in ImageNet class list")

# Step 4: Print and optionally save
print("\n✅ Final matched ImageNet class indices:")
print(matched_indices)

# Optional: save to JSON file
with open("imagenette_target_indices.json", "w") as f:
    json.dump(matched_indices, f, indent=2)
