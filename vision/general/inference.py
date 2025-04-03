import gc
import os
import signal
import traceback
from pathlib import Path

import numpy as np
import torch
from mlperf_loadgen import QuerySampleResponse
from model_loader import ModelLoader
from preprocess import Preprocessor


class ModelPerf:
    def __init__(self, model_name, model_type, dataset_dir, loadgen, preprocess_fn=None, model_architecture=None):
        self.model_name = model_name
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.model_loader = ModelLoader(model_name, model_type=model_type, model_architecture=model_architecture)
        self.dataset_dir = Path(dataset_dir)
        self.preprocess_fn = Preprocessor(model_type, model_name, user_preprocess_fn=preprocess_fn)
        self.loadgen = loadgen
        self.index_to_path = self.create_index_to_path()
        self.samples = {}
        self.device = torch.device("cpu")
        self.inference_indices = set()

        # ─────────────────────────────────────────────────────
        # 3) DISABLE SIGSEGV HANDLER
        # Comment out the custom segfault handler to avoid re-entrant crashes
        # signal.signal(signal.SIGSEGV, self._handle_segfault)
        # ─────────────────────────────────────────────────────

        print("ModelPerf initialized with device:", self.device)

    def _handle_segfault(self, signum, frame):
        """Handle segmentation faults gracefully (DISABLED)."""
        print("CRITICAL: Caught segmentation fault signal! Emergency cleanup in progress...")
        self._emergency_cleanup()
        # Re-raise the signal after cleanup
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGSEGV)

    def _emergency_cleanup(self):
        """Emergency cleanup when segfault is detected."""
        try:
            print("Emergency cleanup: Clearing samples dictionary")
            self.samples.clear()

            print("Emergency cleanup: Clearing CUDA cache")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Emergency cleanup: Running garbage collection")
            gc.collect()

            print("Emergency cleanup complete")
        except Exception as e:
            print(f"Error during emergency cleanup: {e}")

    def create_index_to_path(self):
        """Creates a mapping from indices to dataset sample paths."""
        try:
            image_paths = sorted(self.dataset_dir.glob("*.*"))  # multiple file types
            return {i: str(image_paths[i]) for i in range(len(image_paths))}
        except Exception as e:
            print(f"Error creating index to path mapping: {e}")
            return {}

    def load_query_samples(self, samples):
        """Loads dataset samples into memory after preprocessing."""
        print(f"Loading {len(samples)} samples into memory...")
        for s in samples:
            try:
                if s in self.index_to_path:
                    img_path = self.index_to_path[s]
                    self.samples[s] = self.preprocess_fn.preprocess(img_path)
            except Exception as e:
                print(f"Error loading sample {s}: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Finished loading {len(samples)} samples")

    def issue_queries(self, query_samples):
        """Processes MLPerf queries and runs inference."""
        responses = []
        experiment_name = getattr(self.model_loader, "model_architecture", "default_model")

        try:
            log_dir = Path(f"logs/{experiment_name}")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / "inference_sequence.log"

            batch_inputs = []
            batch_indices = []
            all_responses = []
            total_processed = 0
            countsample=0
            with open(log_file_path, "a") as log_file:
                print(f"Processing {len(query_samples)} query samples...")
                log_file.write(f"Processing {len(query_samples)} query samples\n")

                for i, qs in enumerate(query_samples):
                    try:
                        if qs.index not in self.index_to_path:
                            print(f"Warning: Sample index {qs.index} not found in index_to_path")
                            continue

                        if qs.index not in self.samples:
                            print(f"Warning: Sample {qs.index} not loaded in samples dictionary")
                            continue

                        img_path = self.index_to_path[qs.index]
                        print(f"Image path for sample {qs.index}: {img_path}")
                        input_tensor = self.samples[qs.index]
                        print(f"Input tensor shape for sample {qs.index}: {input_tensor.shape}")
                        countsample+=1
                        print(f"countsample: {countsample}")
                        # Validate
                        if not isinstance(input_tensor, (torch.Tensor, np.ndarray, dict)):
                            print(f"Warning: Sample {qs.index} invalid type: {type(input_tensor)}")
                            continue

                        batch_inputs.append(input_tensor)
                        batch_indices.append(qs)
                        self.inference_indices.add(img_path)

                        # ─────────────────────────────────────────────────────────────────
                        # 1) REDUCE BATCH SIZE
                        # We'll run inference every 8 inputs, not 16 or 32
                        if len(batch_inputs) >= 8:
                            batch_responses = []
                            self._run_batch(batch_inputs, batch_indices, batch_responses)
                            all_responses.extend(batch_responses)
                            total_processed += len(batch_responses)
                            batch_inputs = []
                            batch_indices = []

                            # Force partial results to LoadGen every 1000
                            if total_processed % 1000 == 0:
                                log_file.write(f"Completed {total_processed} samples - sending to loadgen\n")
                                try:
                                    self.loadgen.QuerySamplesComplete(all_responses)
                                    all_responses = []
                                except Exception as e:
                                    print(f"Error sending partial responses to loadgen: {e}")

                            # ─────────────────────────────────────────────────────
                            # Force more frequent GC
                            if i % 200 == 0:  # every 200 queries
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            # ─────────────────────────────────────────────────────

                    except Exception as e:
                        print(f"Error processing sample {qs.index}: {str(e)}")
                        traceback.print_exc()

                # Run remaining
                if batch_inputs:
                    batch_responses = []
                    self._run_batch(batch_inputs, batch_indices, batch_responses)
                    all_responses.extend(batch_responses)

                # Send any leftover responses
                if all_responses:
                    try:
                        self.loadgen.QuerySamplesComplete(all_responses)
                    except Exception as e:
                        print(f"Error sending final responses to loadgen: {e}")

                print(f"Completed all {len(query_samples)} samples")
        except Exception as e:
            print(f"Error in issue_queries: {str(e)}")
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_batch(self, inputs, indices, responses):
        for inp, idx in zip(inputs, indices):
            try:
                if isinstance(inp, np.ndarray):
                    inp = torch.from_numpy(inp)
                inp = inp.to(self.device)
                output = self.model_loader.infer(inp)
                pred = self._get_prediction_from_output(output)
                responses.append(QuerySampleResponse(idx.id, pred, 1))
            except Exception as e:
                print(f"Error with sample {idx.index}: {e}")

    def _get_predictions_from_output(self, outputs, batch_size):
        """Extract predictions from model outputs."""
        try:
            if isinstance(outputs, torch.Tensor):
                return outputs.argmax(dim=1).cpu().tolist()
            elif isinstance(outputs, np.ndarray):
                if outputs.ndim > 1:
                    return outputs.argmax(axis=1).tolist()
                else:
                    return [outputs.argmax()]
            elif isinstance(outputs, list):
                if all(isinstance(o, np.ndarray) for o in outputs):
                    return [o.argmax() for o in outputs]
                else:
                    # Try to recurse
                    return [self._get_prediction_from_output(o) for o in outputs]
            elif hasattr(outputs, "logits"):
                return outputs.logits.argmax(dim=1).cpu().tolist()
            else:
                print(f"Warning: Unhandled output type: {type(outputs)}")
                return [0] * batch_size
        except Exception as e:
            print(f"Error extracting predictions: {e}")
            return [0] * batch_size

    def _get_prediction_from_output(self, output):
        """Extract single prediction from model output."""
        try:
            if isinstance(output, torch.Tensor):
                return output.argmax().item()
            elif isinstance(output, np.ndarray):
                return output.argmax()
            elif isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], np.ndarray):
                    return output[0].argmax()
                return 0
            elif hasattr(output, "logits"):
                return output.logits.argmax().item()
            else:
                return 0
        except Exception as e:
            print(f"Error in _get_prediction_from_output: {e}")
            return 0

    def unload_query_samples(self, samples):
        """Removes samples from memory."""
        print(f"Unloading {len(samples)} samples from memory...")
        try:
            batch_size = 500
            for i in range(0, len(samples), batch_size):
                for s in samples[i : i + batch_size]:
                    if s in self.samples:
                        del self.samples[s]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("Finished unloading samples")
        except Exception as e:
            print(f"Error unloading samples: {e}")
            traceback.print_exc()

    def get_inferred_indices(self):
        """Returns paths of samples that were inferred."""
        return list(self.inference_indices)

    def dataset_size(self):
        """Returns total number of samples."""
        return len(self.index_to_path)

    def flush_queries(self):
        """Required by MLPerf LoadGen interface."""
        print("Flushing queries...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, input_tensor):
        """Helper for accuracy_metrics.py."""
        try:
            output = self.model_loader.infer(input_tensor)
            return self._get_prediction_from_output(output)
        except Exception as e:
            print(f"Error in predict: {e}")
            return 0

    def __del__(self):
        """Cleanup on destruction."""
        print("Destroying ModelPerf, cleaning resources")
        # self.samples.clear()
        # self.inference_indices.clear()
        # gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
