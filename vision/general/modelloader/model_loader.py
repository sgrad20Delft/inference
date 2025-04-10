import importlib
import random
from pathlib import Path

import torch
import onnxruntime as ort
import tensorflow as tf
import torchvision.models as models
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoModelForObjectDetection
from torchvision.transforms.functional import to_pil_image
from transformers import (
        AutoModelForSemanticSegmentation,
        CLIPModel, CLIPProcessor,
        VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    )

import urllib.request

class ModelLoader:
    def __init__(self, model_path, model_type, task_type,input_format="NCHW",model_architecture=None,onnx_postprocess_fn_file=None, tensorflow_postprocess_fn_file=None):
        self.input_format = input_format
        if not model_path or not model_type:
            model_type, model_path = self.get_default_model_for_task(task_type)
        self.task_type = task_type  # Default, will be updated if ONNX detects NHWC
        self.model_path = str(Path(model_path).resolve())
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = self.load_model(model_path, model_type)
        if onnx_postprocess_fn_file:
            spec = importlib.util.spec_from_file_location("onnx_post", onnx_postprocess_fn_file)
            onnx_post_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(onnx_post_module)
            self.onnx_postprocess = getattr(onnx_post_module, "postprocess_onnx_output")
        else:
            self.onnx_postprocess = None
        if tensorflow_postprocess_fn_file:
            spec = importlib.util.spec_from_file_location("tensorflow_post", tensorflow_postprocess_fn_file)
            tensorflow_postprocess_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tensorflow_postprocess_module)
            self.tensorflow_postprocess = getattr(tensorflow_postprocess_module, "tensorflow_postprocess")
        else:
            self.tensorflow_postprocess = None

    import random

    def get_default_model_for_task(self, task_type):
        options = {
            "classification": [
                ("huggingface", "google/vit-base-patch16-224"),
                ("pytorch", "resnet18")
            ],
            "detection": [
                ("huggingface", "facebook/detr-resnet-50"),
                ("pytorch", "fasterrcnn_resnet50_fpn")
            ],
            "segmentation": [
                ("huggingface", "nvidia/segformer-b0-finetuned-ade-512-512"),
                ("pytorch", "deeplabv3_resnet50")
            ],
            "keypoint_detection": [
                ("huggingface", "facebook/detr-resnet-50-keypoint"),
                ("pytorch", "keypointrcnn_resnet50_fpn")
            ],
            "image_captioning": [
                ("huggingface", "nlpconnect/vit-gpt2-image-captioning")
            ],
            "image_retrieval": [
                ("huggingface", "openai/clip-vit-base-patch32")
            ]
        }

        candidates = options.get(task_type, [])
        if not candidates:
            print(f"[WARN] No default model found for task: {task_type}")
            return (None, None)

        chosen = random.choice(candidates)
        print(f"[INFO] Auto-selected default model for task '{task_type}': {chosen[0]} → {chosen[1]}")
        return chosen

    def load_model(self, model_path, model_type):
        if model_type == "pytorch":
            model = self.load_pytorch_model(model_path)
            model.to(self.device)
            model.eval()
            return model

        elif model_type == "onnx":
            print("Loading ONNX model")
            if self.task_type == "segmentation" and model_path.startswith("http"):
                local_file = "deeplabv3_resnet50.onnx"
                urllib.request.urlretrieve(model_path, local_file)
                model_path = local_file

            session = ort.InferenceSession(model_path, providers=[
                "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            ])
            # Detect input format for ONNX
            input_shape = session.get_inputs()[0].shape
            print(f"[INFO] ONNX model input layout detected as: {self.input_format}")
            if input_shape[1] == 3:
                self.input_format = "NCHW"
            elif input_shape[-1] == 3:
                self.input_format = "NHWC"
            else:
                raise ValueError(f"[ERROR] Unsupported input layout: {input_shape}")
            print(f"[INFO] ONNX model input layout detected as: {self.input_format}")
            return session

        elif model_type == "tensorflow":
            print("🔗 Loading TensorFlow model")
            model = tf.saved_model.load(model_path)
            return model.signatures['serving_default']

        elif model_type == "huggingface":
            print("🤗 Loading HuggingFace model")
            if self.task_type == "classification":
                self.processor = AutoImageProcessor.from_pretrained(model_path)
                model = AutoModelForImageClassification.from_pretrained(model_path)

            elif self.task_type == "detection":
                self.processor = AutoImageProcessor.from_pretrained(model_path)
                model = AutoModelForObjectDetection.from_pretrained(model_path)

            elif self.task_type == "segmentation":
                self.processor = AutoImageProcessor.from_pretrained(model_path)
                model = AutoModelForSemanticSegmentation.from_pretrained(model_path)

            elif self.task_type == "keypoint_detection":
                self.processor = AutoImageProcessor.from_pretrained(model_path)
                model = AutoModelForObjectDetection.from_pretrained(model_path)

            elif self.task_type == "image_captioning":
                self.processor = ViTImageProcessor.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = VisionEncoderDecoderModel.from_pretrained(model_path)

            elif self.task_type == "image_retrieval":
                self.processor = CLIPProcessor.from_pretrained(model_path)
                model = CLIPModel.from_pretrained(model_path)

            else:
                raise ValueError(f"Unsupported HuggingFace task: {self.task_type}")

            model.to(self.device)
            model.eval()
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def load_pytorch_model(self, model_path):
        print(f"📦 Loading PyTorch model architecture: {self.model_architecture}")
        if self.model_architecture == "resnet18":
            model = models.resnet18(pretrained=True)
            # num_ftr = model.fc.in_features
            # model.fc = nn.Linear(num_ftr, 10)
        elif self.model_architecture == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
        elif self.model_architecture == "google-net":
            model = models.GoogLeNet
            # Modify classifier for 10 classes
            model.classifier[1] = nn.Linear(
                1280, 10
            )
        elif self.model_architecture == "alexnet":
            model = models.alexnet()
        elif self.model_architecture == "fasterrcnn_resnet50_fpn":
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise ValueError(f"Unsupported PyTorch architecture: {self.model_architecture}")
        print(f"Model loaded successfully!")
        return model

    def infer(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)

            # PyTorch-based inference
            if self.model_type == "pytorch":
                if self.task_type == "classification":
                    if input_tensor.dim() == 3:
                        input_tensor = input_tensor.unsqueeze(0)
                    outputs = self.model(input_tensor)
                    return outputs.cpu()

                elif self.task_type in ["detection", "instance_segmentation", "keypoint_detection"]:
                    inputs = [input_tensor.squeeze(0)]
                    outputs = self.model(inputs)
                    return outputs

                elif self.task_type == "semantic_segmentation":
                    outputs = self.model(input_tensor.unsqueeze(0))
                    return outputs.cpu()

                elif self.task_type == "image_captioning":
                    outputs = self.model.generate(input_tensor)
                    captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
                    return captions

                elif self.task_type == "image_retrieval":
                    outputs = self.model(input_tensor.unsqueeze(0))
                    return outputs.cpu()

            # ONNX-based inference
            elif self.model_type == "onnx":
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = input_tensor.cpu().numpy()


                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: input_tensor})

                if self.task_type == "classification":
                    return torch.tensor(outputs[0])

                elif self.task_type in ["detection", "instance_segmentation", "keypoint_detection","image_captioning"]:
                    # print(f"outputs: {outputs}")
                    if self.onnx_postprocess:
                        return self.onnx_postprocess(outputs, self.task_type)
                    else:
                        return outputs  # raw output if no custom handler


                elif self.task_type == "semantic_segmentation":
                    return torch.tensor(outputs[0])

                elif self.task_type == "image_retrieval":
                    return torch.tensor(outputs[0])

            # TensorFlow-based inference
            elif self.model_type == "tensorflow":
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = input_tensor.cpu().numpy()

                outputs = self.model(input_tensor)

                # TensorFlow outputs are often dict-like; extracting tensors accordingly
                if self.task_type == "classification":
                    return torch.tensor(next(iter(outputs.values())).numpy())

                elif self.task_type in ["detection", "instance_segmentation", "keypoint_detection"]:
                    return {k: v.numpy() for k, v in outputs.items()}

                elif self.task_type == "semantic_segmentation":
                    return torch.tensor(next(iter(outputs.values())).numpy())

                elif self.task_type == "image_captioning":
                    if self.tensorflow_postprocess:
                        return self.tensorflow_postprocess(outputs, self.task_type)
                    else:
                        return outputs

                elif self.task_type == "image_retrieval":
                    return torch.tensor(next(iter(outputs.values())).numpy())

            # Hugging Face-based inference
            elif self.model_type == "huggingface":
                if isinstance(input_tensor, torch.Tensor):
                    images = [to_pil_image(img.cpu()) for img in input_tensor] if input_tensor.ndim == 4 else [
                        to_pil_image(input_tensor.cpu())]
                    inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                else:
                    inputs = {k: v.to(self.device) for k, v in input_tensor.items()}

                if self.task_type == "classification":
                    outputs = self.model(**inputs)
                    return outputs.logits.cpu()

                elif self.task_type == "detection":
                    return self.model(**inputs)

                elif self.task_type == "segmentation":
                    outputs = self.model(**inputs)
                    return outputs.logits.cpu()

                elif self.task_type == "keypoint_detection":
                    return self.model(**inputs)

                elif self.task_type == "image_captioning":
                    pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
                    outputs = self.model.generate(pixel_values)
                    return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                elif self.task_type == "image_retrieval":
                    outputs = self.model.get_image_features(**inputs)
                    return outputs.cpu()

        raise ValueError(f"Unsupported inference type: model_type={self.model_type}, task_type={self.task_type}")


