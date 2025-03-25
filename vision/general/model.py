import torch
import torchvision.models as models

class ModelRunner:
    def __init__(self, model_path=None, model_name=None):
        if model_path:
            self.model = torch.load(model_path)  # Load model from file
        elif model_name:
            self.model = self.load_predefined_model(model_name)
        else:
            raise ValueError("Either model_path or model_name must be provided")

        self.model.eval()

    def load_predefined_model(self, model_name):
        """Load a model by name (e.g., EfficientNet, YOLO)."""
        if model_name.lower() == "efficientnet":
            return models.efficientnet_b0(weights="IMAGENET1K_V1")
        elif model_name.lower() == "resnet":
            return models.resnet50(weights="IMAGENET1K_V1")
        # Add more models here if needed
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def infer(self, input_tensor):
        with torch.no_grad():
            return self.model(input_tensor)