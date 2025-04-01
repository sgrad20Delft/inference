import json
import os
import sys
import argparse
import mlperf_loadgen as lg
import shutil
from pathlib import Path
from inference import ModelPerf  # Generalized model handler

sys.path.append(str(Path(__file__).resolve().parents[2]))


# For dynamic import
import importlib.util
from vision.metrics.loggers_energy.EDE_Cycle import EDECycleCalculator
from vision.metrics.loggers_energy.accuracy_metrics import evaluate_classification_accuracy, evaluate_detection_accuracy
from vision.metrics.loggers_energy.dual_reference_normalizer import DualReferenceNormalizer
from vision.metrics.loggers_energy.unified_logger import UnifiedLogger

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
    print("DEBUG:", custom_module.__file__)
    spec.loader.exec_module(custom_module)
    return custom_module.custom_preprocess  # Ensure the function is named 'custom_preprocess' in the file


def main():
    parser = argparse.ArgumentParser(description="Run MLPerf model evaluation")
    parser.add_argument("--model", type=str, help="Path to the model file or predefined model name.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_type", choices=["pytorch", "onnx", "tensorflow", "huggingface"], help="Specify model type")
    parser.add_argument("--scenario", type=str, default="SingleStream", choices=["SingleStream", "Offline"])
    parser.add_argument("--preprocess_fn_file", type=str, help="File path to custom preprocessing function.")
    parser.add_argument("--task_type", choices=["classification", "detection", "segmentation"], required=True,
                        help="Type of ML task")
    parser.add_argument("--flops", type=int, required=True, help="Model FLOPs (used for EDE scoring)")
    parser.add_argument("--labels_dict", type=str, required=True, help="Path to labels.json file")
    parser.add_argument("--model_architecture",required=True, type=str, default="resnet18",
                        help="Model architecture: resnet18, efficientnet_b0, alexnet, etc.")

    args = parser.parse_args()

    # Dynamically load the custom preprocessing function if provided
    custom_preprocess_fn = None
    if args.preprocess_fn_file:
        custom_preprocess_fn = load_custom_preprocess_fn(args.preprocess_fn_file)

    if args.model_architecture and args.model is None:
        experiment_name = Path(args.model_architecture).stem
    else:
        experiment_name = Path(args.model).stem
    log_base = Path(f"metrics/logs/{experiment_name}")
    log_base.mkdir(parents=True, exist_ok=True)
    (log_base / "codecarbon").mkdir(parents=True, exist_ok=True)
    # Initialize Model Performance Evaluation
    model_perf = ModelPerf(args.model, args.model_type, args.dataset, loadgen=lg, preprocess_fn=custom_preprocess_fn,model_architecture=args.model_architecture )
    with open(args.labels_dict, "r") as f:
        labels_dict = json.load(f)

    # Initialize Unified Energy Logger
    base = Path(__file__).resolve().parent.parent  # goes from general/ to vision/
    print(base)
    rapl_path = base / "metrics/EnergiBridge/target/release/energibridge"
    print(rapl_path)
    energy_logger = UnifiedLogger(
        experiment_name=experiment_name,
        cc_output_dir=f'metrics/logs/{experiment_name}/codecarbon',
        eb_output_file=f'metrics/logs/{experiment_name}/energibridge.csv',
        rapl_power_path=rapl_path
    )

    # === Training Phase ===
    print("⚙️ Training phase with energy tracking:")
    energy_logger.start()
    print("Training...")  # Replace with your actual training logic here
    energy_train = energy_logger.stop()

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
    energy_logger.start()
    lg.StartTest(sut, qsl, settings)
    energy_infer = energy_logger.stop()

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    # Evaluate Accuracy
    if args.task_type == 'classification':
        accuracy, _ = evaluate_classification_accuracy(model_perf, args.dataset, labels_dict)
    elif args.task_type == 'detection':
        accuracy, _ = evaluate_detection_accuracy(model_perf, args.dataset, args.labels_dict)
    else:
        raise NotImplementedError("Segmentation task not implemented yet.")

    # Dual-reference normalization
    normalizer = DualReferenceNormalizer()
    total_energy_wh = energy_train["total_energy_wh"] + energy_infer["total_energy_wh"]
    normalized_energy = normalizer.normalize_energy(total_energy_wh)

    # Compute penalty factor dynamically
    baseline_accuracy = normalizer.refs['accuracy_threshold'][args.task_type]
    penalty_factor = EDECycleCalculator.compute_penalty(accuracy, baseline_accuracy)

    # Compute final EDE score
    ede_score = EDECycleCalculator.compute_ede_cycle(
        accuracy=accuracy,
        flops=args.flops,
        train_energy=energy_train["total_energy_wh"],
        inference_energy=energy_infer["total_energy_wh"],
        alpha=2
    )

    final_ede_score = ede_score * penalty_factor

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
# === Final Comprehensive Benchmark Report ===
    print("Final Benchmark Report:")
    print(f"Accuracy: {accuracy}")
    print(f"Total Energy (Wh): {total_energy_wh}")
    print(f"Normalized Energy: {normalized_energy}")
    print(f"Penalty Factor: {penalty_factor}")
    print(f"Final EDE Score (with penalty): {final_ede_score}")



if __name__ == "__main__":
    main()
