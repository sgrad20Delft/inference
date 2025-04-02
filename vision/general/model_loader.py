
import torch
import onnxruntime as ort
import tensorflow as tf
import torchvision.models as models
from transformers import AutoModelForImageClassification

class ModelLoader:
    def __init__(self, model_path, model_type, model_architecture=None):
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.device=torch.device("cuda:0" if torch.cuda.is_available()else "mps" if torch.backends.mps.is_available() else "cpu")
        print("Device: " + str(self.device))
        self.model = self.load_model(model_path, model_type)

    def load_model(self, model_path, model_type):
        """Dynamically load a model based on its type."""
        print("Got this type: " + model_type)
        if model_type == "pytorch":
            # Use the model architecture passed as an argument or default to resnet18
            model = self.load_pytorch_model(model_path)
            model.eval()  # Set the model to evaluation mode
            print("Model loaded")
            return model
        elif model_type == "onnx":
            return ort.InferenceSession(model_path)
        elif model_type == "tensorflow":
            return self.load_tensorflow_model(model_path)
        elif model_type == "huggingface":
            model=AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
            model.to(self.device)
            model.eval()
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def load_pytorch_model(self, model_path):
        """Dynamically load a PyTorch model based on its architecture."""
        print("Got this architecture: ")
        if self.model_architecture is None:
            self.model_architecture = "resnet18"  # Default architecture
        
        # Dynamically load the specified PyTorch model
        if self.model_architecture == "resnet18":
            model = models.resnet18()
        elif self.model_architecture == "efficientnet_b0":
            print("Efficientnet")
            model = models.efficientnet_b0()
        elif self.model_architecture == "alexnet":
            model = models.alexnet()
        else:
            raise ValueError(f"Unsupported PyTorch model architecture: {self.model_architecture}")
        # model_path = Path(model_path) if model_path else None
        # if model_path is None or not model_path.exists():
        #     print(f"⚠️ Model file '{model_path}' not found. Downloading pretrained weights from torchvision...")
        #     model = models.__dict__[self.model_architecture](pretrained=True)
        #     return model

        # Load the model's state_dict (weights)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)# Load the weights into the model
        return model

    def load_tensorflow_model(self, model_path):
        """Loads a TensorFlow model from a SavedModel format."""
        model = tf.saved_model.load(model_path)
        print("TF Devices:", tf.config.list_physical_devices())
        # Check available signatures and return the default one
        if 'serving_default' not in model.signatures:
            raise ValueError(f"Model does not contain the expected 'serving_default' signature.")
        
        return model.signatures['serving_default']  # This returns the inference function


    # #def preprocess(self, input_data):
    #     """Handle preprocessing based on model type.
    #     if self.model_type == "pytorch" or self.model_type == "onnx":
    #         return torch.tensor(input_data).float()
    #     elif self.model_type == "tensorflow":
    #         return tf.convert_to_tensor(input_data, dtype=tf.float32)
    #     elif self.model_type == "huggingface":
    #         feature_extractor = ViTFeatureExtractor.from_pretrained(self.model)
    #         return feature_extractor(input_data, return_tensors="pt")
    #     else:
    #         raise ValueError("Unknown preprocessing method")"""
        
    def infer(self, input_tensor):
        """Runs inference on the model."""
        if self.model_type == "pytorch":
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                return self.model(input_tensor)
        elif self.model_type == "onnx":
            if hasattr(input_tensor, "numpy"):  # If it's a PyTorch tensor
                input_tensor = input_tensor.numpy()
            return self.model.run(None, {self.model.get_inputs()[0].name: input_tensor})
        elif self.model_type == "tensorflow":
            return self.model(input_tensor)
        elif self.model_type == "huggingface":
            return self.model(**input_tensor)
        else:
            raise ValueError("Unsupported inference type")