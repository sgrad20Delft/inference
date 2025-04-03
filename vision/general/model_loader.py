import torch
import onnxruntime as ort
import tensorflow as tf
import torchvision.models as models
from torch import nn
from transformers import AutoModelForImageClassification, AutoImageProcessor


class ModelLoader:
    def __init__(self, model_path, model_type, model_architecture=None):
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = self.load_model(model_path, model_type)

    def load_model(self, model_path, model_type):
        if model_type == "pytorch":
            model = self.load_pytorch_model(model_path)
            model.to(self.device)
            model.eval()
            return model

        elif model_type == "onnx":
            print("Loading ONNX model")
            return ort.InferenceSession(model_path, providers=[
                "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"])

        elif model_type == "tensorflow":
            print("🔗 Loading TensorFlow model")
            model = tf.saved_model.load(model_path)
            return model.signatures['serving_default']

        elif model_type == "huggingface":
            print("🤗 Loading HuggingFace model")
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            return model

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def load_pytorch_model(self, model_path):
        print(f"📦 Loading PyTorch model architecture: {self.model_architecture}")
        if self.model_architecture == "resnet18":
            model = models.resnet18()
        elif self.model_architecture == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            print("Required Model")
            # model.classifier[1]=nn.Linear(model.classifier[1], 10)
        elif self.model_architecture == "alexnet":
            model = models.alexnet()
        else:
            raise ValueError(f"Unsupported PyTorch architecture: {self.model_architecture}")

        # state_dict = torch.load(model_path, map_location=self.device)
        # model.classifier[1] = nn.Linear(1280, 10)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict,strict=False)
        return model

    def infer(self, input_tensor):
        if self.model_type == "pytorch":
            self.model.eval()
            input_tensor = input_tensor.to("cpu")
            # print("DEBUG: final input_tensor shape =", input_tensor.shape)
            if input_tensor.dim() == 5 and input_tensor.shape[1] == 1:
                input_tensor = input_tensor.squeeze(1)
            elif input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dim

            input_tensor = input_tensor.float()

            with torch.no_grad():
                try:
                    output = self.model(input_tensor.to(self.device))
                    return output.cpu()  # Return tensor instead of numpy for flexibility
                except Exception as e:
                    print(f"Error during model inference: {e}")
                    raise

        elif self.model_type == "onnx":
            try:
                # Handle 5D tensor if present
                if isinstance(input_tensor, torch.Tensor):
                    if input_tensor.dim() == 5 and input_tensor.shape[1] == 1:
                        input_tensor = input_tensor.squeeze(1)

                    if hasattr(input_tensor, "numpy"):
                        input_tensor = input_tensor.detach().cpu().numpy()

                input_name = self.model.get_inputs()[0].name
                return self.model.run(None, {input_name: input_tensor})
            except Exception as e:
                print(f"Error during ONNX inference: {e}")
                raise

        elif self.model_type == "tensorflow":
            try:
                # Handle 5D tensor if present
                if isinstance(input_tensor, torch.Tensor):
                    if input_tensor.dim() == 5 and input_tensor.shape[1] == 1:
                        input_tensor = input_tensor.squeeze(1)

                    if hasattr(input_tensor, "numpy"):
                        input_tensor = input_tensor.detach().cpu().numpy()

                return self.model(input_tensor)
            except Exception as e:
                print(f"Error during TensorFlow inference: {e}")
                raise

        elif self.model_type == "huggingface":
            try:
                if isinstance(input_tensor, torch.Tensor):
                    # Handle 5D tensor if present
                    if input_tensor.dim() == 5 and input_tensor.shape[1] == 1:
                        input_tensor = input_tensor.squeeze(1)

                    inputs = self.processor(images=input_tensor, return_tensors="pt").to(self.device)
                else:
                    inputs = {k: v.to(self.device) for k, v in input_tensor.items()}

                with torch.no_grad():
                    return self.model(**inputs)
            except Exception as e:
                print(f"Error during HuggingFace inference: {e}")
                raise

        else:
            raise ValueError("Unsupported inference type")