import json
from pathlib import Path
from vision.general.modelloader.model_loader import  ModelLoader  # Import ModelLoader
from vision.general.datasetpreprocess.preprocess import Preprocessor
from mlperf_loadgen import QuerySampleResponse
import torch
import time

import numpy as np
class ModelPerf:
    def __init__(self, model_name, model_type, dataset_dir, model_architecture,loadgen,task_type, preprocess_fn=None,
                 onnx_postprocess_fn=None, tensorflow_postprocess_fn=None):
        """
        MLPerf benchmarking system using ModelLoader.

        Args:
            model_path (str): Path to the model file or Hugging Face model name.
            dataset_dir (str): Path to dataset directory.
            preprocess_fn (callable): User-provided preprocessing function.
            loadgen: MLPerf LoadGen instance.
            model_type (str, optional): Specify "pytorch", "onnx", "tensorflow", or "huggingface".
        """
        self.task_type = task_type
        self.model_loader = ModelLoader(model_name, model_type=model_type,model_architecture=model_architecture,task_type=task_type,
                                        onnx_postprocess_fn_file=onnx_postprocess_fn, tensorflow_postprocess_fn_file=tensorflow_postprocess_fn)
        self.dataset_dir = Path(dataset_dir)
        print(f"Dataset directory: {self.dataset_dir}")
        self.preprocess_fn = Preprocessor(model_type, model_name, user_preprocess_fn=preprocess_fn,layout=self.model_loader.input_format)
        self.model_architecture = model_architecture
        self.loadgen = loadgen
        self.index_to_path = self.create_index_to_path()
        self.samples = {}
        self.label_index_map = self.create_label_for_imagenet()
        self.predictions_buffer = []
        self.predictions_log = {}
        self.latencies=[]

    def check_label_index_map(self, prediction):
        # Filter to 10-class subset
       if self.label_index_map and prediction in self.label_index_map:
            prediction = self.label_index_map[prediction]
            print(f"[INFO] Matching prediction: {prediction}")
       else:  # Skip predictions outside your 10-class subset
            print(f"[WARNING] Skipping unmatched prediction: {prediction}")
            prediction = 999
       return prediction


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
        self.latencies = []
        countsample=0
        for qs in query_samples:
            start_time = time.time()
            input_tensor = self.samples[qs.index]
            output = self.model_loader.infer(input_tensor)
            countsample+=1
            filename = Path(self.index_to_path[qs.index]).name
            print(f"countsample: {countsample}")
            # Classification Task
            if self.task_type == "classification":
                if isinstance(output, torch.Tensor):
                    prediction = output.argmax().item()
                elif isinstance(output, list):
                    prediction = np.array(output[0]).argmax()
                elif hasattr(output, "logits"):
                    prediction = output.logits.argmax().item()
                else:
                    raise ValueError("Unsupported model output type for classification.")

                if self.label_index_map:
                    prediction = self.check_label_index_map(prediction)

                pred_np = np.array([prediction], dtype=np.int32)
                self.predictions_log[filename] = prediction

            # Detection / Instance Segmentation / Keypoint Detection Task
            elif self.task_type in ["detection", "instance_segmentation", "keypoint_detection"]:
                predictions = output[0] if isinstance(output, list) and len(output)!=0 else output
                print(f"predictions: {predictions}")
                # formatted_output = {
                #     "boxes": predictions["boxes"].cpu().tolist() if "boxes" in predictions else [],
                #     "scores": predictions["scores"].cpu().tolist() if "scores" in predictions else [],
                #     "labels": predictions["labels"].cpu().tolist() if "labels" in predictions else [],
                #     "keypoints": predictions["keypoints"].cpu().tolist() if "keypoints" in predictions else [],
                #     "masks": predictions["masks"].cpu().numpy().tolist() if "masks" in predictions else []
                # }
                formatted_output = {}
                if hasattr(predictions, 'items'):
                    for key, value in predictions.items():
                        if isinstance(value, torch.Tensor):
                            formatted_output[key] = value.cpu().tolist()
                        elif isinstance(value, np.ndarray):
                            formatted_output[key] = value.tolist()
                        else:
                            formatted_output[key] = value
                else:
                    formatted_output = predictions  # fallback for non-dict predictions

                print(f"formatted_output: {formatted_output}")
                self.predictions_log[filename] = formatted_output
                pred_np = np.array([0], dtype=np.int32)  # Dummy prediction for MLPerf compliance

            # Semantic Segmentation Task
            elif self.task_type == "semantic_segmentation":
                if isinstance(output, torch.Tensor):
                    mask_array = output.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
                elif isinstance(output, np.ndarray):
                    mask_array = np.argmax(output, axis=1).squeeze(0)
                else:
                    raise ValueError("Unsupported semantic segmentation output type.")

                self.predictions_log[filename] = mask_array.tolist()
                pred_np = np.array([0], dtype=np.int32)  # Dummy prediction

            # Image Captioning Task
            elif self.task_type == "image_captioning":
                caption = output[0] if isinstance(output, list) else output
                self.predictions_log[filename] = caption
                pred_np = np.array([0], dtype=np.int32)  # Dummy prediction

            # Image Retrieval Task
            elif self.task_type == "image_retrieval":
                embedding = output.squeeze().cpu().numpy().tolist() if isinstance(output, torch.Tensor) else np.array(
                    output).tolist()
                self.predictions_log[filename] = embedding
                pred_np = np.array([0], dtype=np.int32)  # Dummy prediction

            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self.latencies.append(latency_ms)

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
