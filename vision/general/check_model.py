import torch
from torchvision import models
from torch import nn

# Define the model architecture (10 output classes)
model = models.efficientnet_b0(pretrained=False)  # Start with NO pretrained weights
model.classifier[1] = nn.Linear(
    in_features=model.classifier[1].in_features,  # Keep original in_features (1280)
    out_features=10  # Modify out_features for MNIST (10 classes)
)

# Load your checkpoint
checkpoint = torch.load("models/efficientnet_b0.pth", map_location="cpu")

# Attempt to load state_dict into the model
try:
    model.load_state_dict(checkpoint)
    print("✅ Checkpoint loaded successfully! Model has 10 output classes.")
except RuntimeError as e:
    print("❌ Checkpoint mismatch! Error:", e)

# Explicitly check classifier shape
print("\nClassifier layer weights shape:", model.classifier[1].weight.shape)
print("Classifier layer bias shape:", model.classifier[1].bias.shape)