from mlperf_loadgen import QuerySampleResponse
from vision.efficientnet.model import EfficientNetRunner
from vision.efficientnet.preprocess import preprocess
from pathlib import Path
import torch

class EfficientNetMLPerf:
    def __init__(self, loadgen):
        self.model = EfficientNetRunner()
        self.dataset_dir = "mnist_images"
        self.index_to_path = self.create_index_to_path()
        self.samples = {}
        self.loadgen = loadgen

    def create_index_to_path(self):
        """Creates a mapping from sample indices to MNIST image file paths."""
        image_paths = sorted(Path(self.dataset_dir).glob("*.png"))
        return {i: str(image_paths[i]) for i in range(len(image_paths))}

    def load_query_samples(self, samples):
        for s in samples:
            img_path = self.index_to_path[s]
            self.samples[s] = preprocess(img_path)

    def issue_queries(self, query_samples):
        responses = []
        for qs in query_samples:
            input_tensor = self.samples[qs.index]
            output = self.model.infer(input_tensor)
            prediction = torch.argmax(output).item()
            response = QuerySampleResponse(qs.id, prediction, 1)
            responses.append(response)
        
        # Now report all responses at once
        self.loadgen.QuerySamplesComplete(responses)

    def unload_query_samples(self, samples):
        for s in samples:
            del self.samples[s]

    def flush_queries(self):
        pass