from pathlib import Path
from model_loader import ModelLoader  # Import ModelLoader
from preprocess import Preprocessor
from mlperf_loadgen import QuerySampleResponse
import torch

class ModelPerf:
    def __init__(self, model_name, model_type, dataset_dir, loadgen, preprocess_fn=None,model_architecture=None):
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
        self.preprocess_fn = Preprocessor(model_type, model_name, user_preprocess_fn=preprocess_fn)
        self.loadgen = loadgen
        self.index_to_path = self.create_index_to_path()
        self.samples = {}

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
        experiment_name = getattr(self.model_loader, "model_architecture", "default_model")
        log_dir = Path(f"logs/{experiment_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "inference_sequence.log"
        with open(log_file_path, "a") as log_file:
         for qs in query_samples:
            input_tensor = self.samples[qs.index]
            image_path = self.index_to_path[qs.index]
            print(f"🔍 Inference on sample ID: {qs.index}, file: {image_path}")
            log_file.write(f"Inference on sample ID: {qs.index}, file: {image_path}\n")
            output = self.model_loader.infer(input_tensor)

            # Generalized response handling
            if isinstance(output, torch.Tensor):
                prediction = output.argmax().item()
            elif isinstance(output, list):  # ONNX or TensorFlow returns lists
                prediction = output[0].argmax()
            elif hasattr(output, "logits"):  # Hugging Face ImageClassifierOutput
                prediction = output.logits.argmax().item()
            else:
                raise ValueError("Unsupported model output type.")

            response = QuerySampleResponse(qs.id, prediction, 1)
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