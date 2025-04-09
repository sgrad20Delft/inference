import json
import urllib.request

# Load the mapping from ImageNet index to (WNID, label)
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    imagenet_index_map = json.load(response)

# Get WNIDs for your indices
target_indices = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
index_to_wnid = {int(k): v[0] for k, v in imagenet_index_map.items()}

# Print matching WNIDs
for idx in target_indices:
    print(f"{index_to_wnid[idx]}: {idx}")
