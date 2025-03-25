from mlperf_loadgen import QuerySampleResponse
from pathlib import Path
import torch

class ModelPerf:
    def __init__(self, model_path_or_name, dataset_dir, preprocess_fn, loadgen, model_cls=None):
        self.dataset_dir = dataset_dir
        self.index_to_path = self.create_index_to_path()
        self.samples = {}
        self.preprocess_fn = preprocess_fn  # Custom preprocessing function
        self.loadgen = loadgen

        # Load model dynamically
        if model_cls:
            self.model = model_cls()
        else:
            self.model = self.load_model(model_path_or_name)

    def load_model(self, model_path_or_name):
        """Loads a model dynamically from a given path or name."""
        if model_path_or_name == "efficientnet":
            from vision.efficientnet.inference import EfficientNetRunner
            return EfficientNetRunner()
        elif model_path_or_name == "yolo":
            from vision.yolo.inference import YOLORunner
            return YOLORunner()
        else:
            return torch.jit.load(model_path_or_name)  # Load a TorchScript model from file

    def create_index_to_path(self):
        """Creates a mapping from sample indices to image file paths."""
        image_paths = sorted(Path(self.dataset_dir).glob("*.*"))  # Support multiple formats
        return {i: str(image_paths[i]) for i in range(len(image_paths))}

    def load_query_samples(self, samples):
        for s in samples:
            img_path = self.index_to_path[s]
            self.samples[s] = self.preprocess_fn(img_path)

    def issue_queries(self, query_samples):
        responses = []
        for qs in query_samples:
            input_tensor = self.samples[qs.index]
            output = self.model.infer(input_tensor)

            # Generalized response handling
            if hasattr(output, 'xyxy'):  # Object detection (e.g., YOLO)
                prediction = len(output.xyxy[0])
            else:  # Classification (e.g., EfficientNet)
                prediction = output.argmax().item()

            response = QuerySampleResponse(qs.id, prediction, 1)
            responses.append(response)

        self.loadgen.QuerySamplesComplete(responses)

    def unload_query_samples(self, samples):
        for s in samples:
            del self.samples[s]

    def dataset_size(self):
        """Returns the total number of samples in the dataset."""
        return len(self.index_to_path)

    def flush_queries(self):
        pass