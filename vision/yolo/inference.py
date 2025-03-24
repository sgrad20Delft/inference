from mlperf_loadgen import QuerySampleResponse
from vision.yolo.model import YOLORunner
from pathlib import Path
from vision.yolo.preprocess import preprocess

class YOLOMLPerf:
    def __init__(self):
        self.model = YOLORunner()
        self.dataset_dir = "mnist_images"
        self.index_to_path = self.create_index_to_path()
        self.samples = {}

    def create_index_to_path(self):
        """Creates a mapping from sample indices to MNIST image file paths."""
        image_paths = sorted(Path(self.dataset_dir).glob("*.png"))
        return {i: str(image_paths[i]) for i in range(len(image_paths))}

    def load_query_samples(self, samples):
        for s in samples:
            img_path = self.index_to_path[s]
            self.samples[s] = preprocess(img_path)

    def issue_queries(self, query_samples):
        for qs in query_samples:
            input_tensor = self.samples[qs.index]
            output = self.model.infer(input_tensor)
            prediction = len(output.xyxy[0])  # Just example: number of detections
            response = QuerySampleResponse(qs.id, prediction, 1)
            QuerySampleResponse.enqueue([response])

    def unload_query_samples(self, samples):
        for s in samples:
            del self.samples[s.index]

    def flush_queries(self):
        pass
