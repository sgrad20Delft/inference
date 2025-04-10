import torch

class YOLORunner:
    def __init__(self, model_path="yolov5s.pt"):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.eval()

    def infer(self, input_tensor):
        with torch.no_grad():
            return self.model(input_tensor)
