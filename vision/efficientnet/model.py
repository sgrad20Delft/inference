import torch
import torchvision.models as models

class EfficientNetRunner:
    def __init__(self):
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.model.eval()

    def infer(self, input_tensor):
        with torch.no_grad():
            return self.model(input_tensor)