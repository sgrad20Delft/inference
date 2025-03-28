import os
import sys
import argparse
import mlperf_loadgen as lg
import shutil
from pathlib import Path
from inference import ModelPerf  # Generalized model handler

# For dynamic import
import importlib.util

sys.path.append(os.path.abspath("."))

def load_custom_preprocess_fn(filepath):
    """
    Dynamically loads a custom preprocessing function from a Python file.

    Args:
        filepath (str): Path to the Python file containing the custom preprocessing function.

    Returns:
        callable: The custom preprocessing function.
    """
    spec = importlib.util.spec_from_file_location("custom_preprocess", filepath)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    return custom_module.custom_preprocess  # Ensure the function is named 'custom_preprocess' in the file


def main():
    parser = argparse.ArgumentParser(description="Run MLPerf model evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file or predefined model name.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_type", choices=["pytorch", "onnx", "tensorflow", "huggingface"], help="Specify model type")
    parser.add_argument("--scenario", type=str, default="SingleStream", choices=["SingleStream", "Offline"])
    parser.add_argument("--preprocess_fn_file", type=str, help="File path to custom preprocessing function.")
    args = parser.parse_args()

    # Dynamically load the custom preprocessing function if provided
    custom_preprocess_fn = None
    if args.preprocess_fn_file:
        custom_preprocess_fn = load_custom_preprocess_fn(args.preprocess_fn_file)

    # Initialize Model Performance Evaluation
    model_perf = ModelPerf(args.model, args.model_type, args.dataset, loadgen=lg, preprocess_fn=custom_preprocess_fn)

    # LoadGen config
    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "Offline": lg.TestScenario.Offline
    }

    log_path = Path(f"./mlperf_{Path(args.model).stem}_log")
    os.makedirs(log_path, exist_ok=True)

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.mode = lg.TestMode.PerformanceOnly

    sut = lg.ConstructSUT(
        model_perf.issue_queries,
        model_perf.flush_queries
    )
    qsl = lg.ConstructQSL(
        model_perf.dataset_size(),
        model_perf.dataset_size(),
        model_perf.load_query_samples,
        model_perf.unload_query_samples
    )

    print(f"Running MLPerf Inference for model: {args.model} with dataset: {args.dataset}")
    lg.StartTest(sut, qsl, settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    # List of MLPerf log files
    log_files = [
        "mlperf_log_accuracy.json",
        "mlperf_log_detail.txt",
        "mlperf_log_summary.txt",
        "mlperf_log_trace.json"
    ]

    # Move each log file if it exists
    for log_file in log_files:
        src = Path(log_file)
        dest = log_path / log_file  # Destination in the log directory

        if src.exists():
            shutil.move(str(src), str(dest))  # Move the file
            print(f"Moved {src} -> {dest}")
        else:
            print(f"File {src} not found, skipping.")


if __name__ == "__main__":
    main()