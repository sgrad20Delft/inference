import torch
import onnxruntime as ort
import tensorflow as tf
import numpy as np
import torchvision.models as models
from transformers import AutoModel, ViTFeatureExtractor , AutoModelForImageClassification

class ModelLoader:
    def __init__(self, model_path, model_type, model_architecture=None):
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.model = self.load_model(model_path, model_type)

    def load_model(self, model_path, model_type):
        """Dynamically load a model based on its type."""
        print("Got this type: " + model_type)
        if model_type == "pytorch":
            # Use the model architecture passed as an argument or default to resnet18
            model = self.load_pytorch_model(model_path)
            model.eval()  # Set the model to evaluation mode
            return model
        elif model_type == "onnx":
            return ort.InferenceSession(model_path)
        elif model_type == "tensorflow":
            return self.load_tensorflow_model(model_path)
        elif model_type == "huggingface":
            return AutoModelForImageClassification.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def load_pytorch_model(self, model_path):
        """Dynamically load a PyTorch model based on its architecture."""
        if self.model_architecture is None:
            self.model_architecture = "resnet18"  # Default architecture
        
        # Dynamically load the specified PyTorch model
        if self.model_architecture == "resnet18":
            model = models.resnet18()
        elif self.model_architecture == "efficientnet_b0":
            model = models.efficientnet_b0()
        elif self.model_architecture == "alexnet":
            model = models.alexnet()
        else:
            raise ValueError(f"Unsupported PyTorch model architecture: {self.model_architecture}")

        # Load the model's state_dict (weights)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)  # Load the weights into the model
        return model

    def load_tensorflow_model(self, model_path):
        """Loads a TensorFlow model from a SavedModel format."""
        model = tf.saved_model.load(model_path)
        
        # Check available signatures and return the default one
        if 'serving_default' not in model.signatures:
            raise ValueError(f"Model does not contain the expected 'serving_default' signature.")
        
        return model.signatures['serving_default']  # This returns the inference function

    def preprocess(self, input_data):
        """Handle preprocessing based on model type."""
        if self.model_type == "pytorch" or self.model_type == "onnx":
            return torch.tensor(input_data).float()
        elif self.model_type == "tensorflow":
            return tf.convert_to_tensor(input_data, dtype=tf.float32)
        elif self.model_type == "huggingface":
            feature_extractor = ViTFeatureExtractor.from_pretrained(self.model)
            return feature_extractor(input_data, return_tensors="pt")
        else:
            raise ValueError("Unknown preprocessing method")
        
    def infer(self, input_tensor):
        """Runs inference on the model."""
        if self.model_type == "pytorch":
            with torch.no_grad():
                return self.model(input_tensor)
        elif self.model_type == "onnx":
            return self.model.run(None, {self.model.get_inputs()[0].name: input_tensor.numpy()})
        elif self.model_type == "tensorflow":
            return self.model(input_tensor)
        elif self.model_type == "huggingface":
            return self.model(**input_tensor)
        else:
            raise ValueError("Unsupported inference type")