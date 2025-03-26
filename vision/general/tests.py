from model_loader import ModelLoader
import torch
import numpy as np

pytorch_model_path = 'D:\\Program Files (x86)\\inference\\vision\\general\\models\\resnet18-f37072fd.pth'
onnx_model_path = 'D:\\Program Files (x86)\\inference\\vision\\general\\models\\resnet18_Opset16.onnx'
tensorflow_model_path = 'D:\\Program Files (x86)\\inference\\vision\\general\\models\\tensor_flow_model'
huggingface_model_path = 'google/vit-base-patch16-224-in21k' # Hugging Face model

# Example test input data
input_data = np.random.rand(1, 3, 224, 224)

# Initialize the model loader
# Test for PyTorch
model_loader_pytorch = ModelLoader(pytorch_model_path, model_type="pytorch")
input_tensor_pytorch = model_loader_pytorch.preprocess(input_data)
output_pytorch = model_loader_pytorch.infer(input_tensor_pytorch)
print("PyTorch Model Output:", output_pytorch)

# Test for ONNX
model_loader_onnx = ModelLoader(onnx_model_path, model_type="onnx")
input_tensor_onnx = model_loader_onnx.preprocess(input_data)
output_onnx = model_loader_onnx.infer(input_tensor_onnx)
print("ONNX Model Output:", output_onnx)

# Test for TensorFlow
"""
model_loader_tensorflow = ModelLoader(tensorflow_model_path, model_type="tensorflow")
input_tensor_tensorflow = model_loader_tensorflow.preprocess(input_data)
output_tensorflow = model_loader_tensorflow.infer(input_tensor_tensorflow)
print("TensorFlow Model Output:", output_tensorflow)"""

# Test for HuggingFace
"""
model_loader_huggingface = ModelLoader(huggingface_model_path, model_type="huggingface")
input_tensor_huggingface = model_loader_huggingface.preprocess(input_data)
output_huggingface = model_loader_huggingface.infer(input_tensor_huggingface)
print("HuggingFace Model Output:", output_huggingface)"""