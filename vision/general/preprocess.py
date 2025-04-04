import torch
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import AutoImageProcessor
from torchvision import transforms

transform = transforms.Compose([
            transforms.Resize((224, 224)),               # Resize to 224x224
            transforms.Grayscale(num_output_channels=3), # Convert 1-channel to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
def custom_preprocess(image_path):
    # print(f"[CUSTOM PREPROCESS] Handling image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    tensor=transform(image).unsqueeze(0)
    # print("[CUSTOM PREPROCESS] shape:", tensor.shape)
    return tensor


class Preprocessor:
    def __init__(self, model_type, model_name, user_preprocess_fn=None, input_size=(224, 224)):
        """
        Handles input preprocessing based on model type.
        
        Args:
            model_type (str): Model format ("pytorch", "onnx", "tensorflow", "huggingface").
            user_preprocess_fn (callable, optional): Custom preprocessing function.
            input_size (tuple): Target image size (default 224x224).
        """
        self.model_name = model_name
        self.model_type = model_type
        self.user_preprocess_fn = user_preprocess_fn
        self.input_size = input_size  # Resize all images to a fixed size


    def preprocess(self, image_path):
        """
        Preprocess an image file based on model type.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            Processed tensor/array depending on model type.
        """
        if self.user_preprocess_fn:
            return self.user_preprocess_fn(image_path)  # Use custom function if provided

        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        image = image.resize(self.input_size)  # Resize image
        
        if self.model_type == "pytorch":
            return self._preprocess_pytorch(image)
        elif self.model_type == "onnx":
            return self._preprocess_onnx(image)
        elif self.model_type == "tensorflow":
            return self._preprocess_tensorflow(image)
        elif self.model_type == "huggingface":
            return self._preprocess_huggingface(image)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _preprocess_pytorch(self, image):
        """Preprocess for PyTorch models (convert to tensor, normalize)."""
        transform_val= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform_val(image).unsqueeze(0) # Add batch dimension

    def _preprocess_onnx(self, image):
        """Preprocess for ONNX models (convert to NumPy array, normalize)."""
        image = np.array(image).astype(np.float32) / 255.0  # Normalize
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        return np.expand_dims(image, axis=0)  # Add batch dimension

    def _preprocess_tensorflow(self, image):
        """Preprocess for TensorFlow models (convert to tensor, normalize)."""
        image = np.array(image).astype(np.float32) / 255.0  # Normalize
        return tf.convert_to_tensor(np.expand_dims(image, axis=0))  # Add batch dim

    def _preprocess_huggingface(self, image):
        """Preprocess for Hugging Face models (use Hugging Face tokenizer)."""
        extractor = AutoImageProcessor.from_pretrained(self.model_name)
        return extractor(image, return_tensors="pt")  # Convert to tensor