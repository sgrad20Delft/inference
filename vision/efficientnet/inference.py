from mlperf_loadgen import QuerySampleResponse
from vision.efficientnet.model import EfficientNetRunner
from vision.efficientnet.preprocess import preprocess
import torch

class EfficientNetMLPerf:
    def __init__(self):
        self.model = EfficientNetRunner()
        self.samples = {}

    def load_query_samples(self, samples):
        for s in samples:
            img_path = s.index
            self.samples[s.index] = preprocess(img_path)

    def issue_queries(self, query_samples):
        for qs in query_samples:
            input_tensor = self.samples[qs.index]
            output = self.model.infer(input_tensor)
            prediction = torch.argmax(output).item()
            response = QuerySampleResponse(qs.id, prediction, 1)
            QuerySampleResponse.enqueue([response])

    def unload_query_samples(self, samples):
        for s in samples:
            del self.samples[s.index]

    def flush_queries(self):
        pass