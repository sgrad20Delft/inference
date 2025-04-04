import json
from pathlib import Path
from model_loader import ModelLoader  # Import ModelLoader
from preprocess import Preprocessor
from mlperf_loadgen import QuerySampleResponse
import torch
import numpy as np
class ModelPerf:
    def __init__(self, model_name, model_type, dataset_dir, model_architecture,loadgen, preprocess_fn=None):
        """
        MLPerf benchmarking system using ModelLoader.

        Args:
            model_path (str): Path to the model file or Hugging Face model name.
            dataset_dir (str): Path to dataset directory.
            preprocess_fn (callable): User-provided preprocessing function.
            loadgen: MLPerf LoadGen instance.
            model_type (str, optional): Specify "pytorch", "onnx", "tensorflow", or "huggingface".
        """

        self.model_loader = ModelLoader(model_name, model_type=model_type,model_architecture=model_architecture)
        self.dataset_dir = Path(dataset_dir)
        print(f"Dataset directory: {self.dataset_dir}")
        self.preprocess_fn = Preprocessor(model_type, model_name, user_preprocess_fn=preprocess_fn)
        self.model_architecture = model_architecture
        self.loadgen = loadgen
        self.index_to_path = self.create_index_to_path()
        self.samples = {}
        self.label_index_map = self.create_label_for_imagenet()


    def create_label_for_imagenet(self):
        print(f"self.dataset_dir: {self.dataset_dir}")
        if str(self.dataset_dir).endswith("imagenette/imagenette_images"):
            # Load custom class index mapping from config
            print("Loading custom class index mapping...")
            with open("vision/dataset_dir/imagenette/labels/imagenette_10_class_map.json") as f:
                return {int(k): v for k, v in json.load(f).items()}

        else:
            print("No custom class index mapping found.")
            return None
    def create_index_to_path(self):
        """Creates a mapping from indices to dataset sample paths."""
        image_paths = sorted(self.dataset_dir.glob("*.*"))  # Supports multiple file types
        return {i: str(image_paths[i]) for i in range(len(image_paths))}

    def load_query_samples(self, samples):
        """Loads dataset samples into memory after preprocessing."""
        for s in samples:
            img_path = self.index_to_path[s]
            self.samples[s] = self.preprocess_fn.preprocess(img_path)

    def issue_queries(self, query_samples):
        """Processes MLPerf queries and runs inference."""
        responses = []
        self.predictions_buffer = []
        # Keep refs to prediction buffers
        countsample=0
        for qs in query_samples:
            input_tensor = self.samples[qs.index]
            output = self.model_loader.infer(input_tensor)
            countsample+=1
            print(f"Processing sample {countsample}")
            if isinstance(output, torch.Tensor):
                prediction = output.argmax().item()
            elif isinstance(output, list):  # ONNX or TensorFlow returns lists
                prediction = output[0].argmax()
            elif hasattr(output, "logits"):
                prediction = output.logits.argmax().item()
            else:
                raise ValueError("Unsupported model output type.")

            # Filter to 10-class subset
            if self.label_index_map and prediction in self.label_index_map:
                prediction = self.label_index_map[prediction]
            else:
                print(f"[WARNING] Skipping unmatched prediction: {prediction}")
                continue  # Skip predictions outside your 10-class subset

            pred_np = np.array([prediction], dtype=np.int32)
            self.predictions_buffer.append(pred_np)

            response = QuerySampleResponse(qs.id, pred_np.ctypes.data, pred_np.nbytes)
            responses.append(response)

        self.loadgen.QuerySamplesComplete(responses)

    def unload_query_samples(self, samples):
        """Removes samples from memory."""
        for s in samples:
            if s in self.samples:
                del self.samples[s]

    def dataset_size(self):
        """Returns total number of samples in dataset."""
        return len(self.index_to_path)


    def flush_queries(self):
        pass

    def predict(self, input_tensor):
        """Batch prediction using the model loader."""
        with torch.no_grad():
            output = self.model_loader.infer(input_tensor)
            if isinstance(output, torch.Tensor):
                preds = output.argmax(dim=1).cpu().numpy().tolist()
            elif isinstance(output, list):
                preds = [o.argmax() for o in output]
            elif hasattr(output, "logits"):
                preds = output.logits.argmax(dim=1).cpu().numpy().tolist()
            else:
                raise ValueError("Unsupported model output type for batch inference.")
        return preds
