import csv
import json
import os
import platform
import random
import sys
import argparse
from datetime import datetime
from time import sleep
import gc
import traceback

import importlib.util

import mlperf_loadgen as lg
import shutil
from pathlib import Path
import subprocess
from PIL import Image

from inference import ModelPerf  # Generalized model handler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from vision.metrics.loggers_energy.EDE_Cycle import EDECycleCalculator
from vision.metrics.loggers_energy.accuracy_metrics import evaluate_classification_accuracy, evaluate_detection_accuracy
from vision.metrics.loggers_energy.dual_reference_normalizer import DualReferenceNormalizer
from vision.metrics.loggers_energy.unified_logger import UnifiedLogger
from vision.metrics.loggers_energy.accuracy_metrics import evaluate_classification_accuracy_from_dict

sys.path.append(os.path.abspath("."))


def dataset_size(self):
    """Returns total number of samples in the dataset."""
    return len(self.index_to_path)


def load_custom_preprocess_fn(filepath):
    """
    Dynamically loads a custom preprocessing function from a Python file.

    Args:
        filepath (str): Path to the Python file containing the custom preprocessing function.

    Returns:
        callable: The custom preprocessing function.
    """
    spec = importlib.util.spec_from_file_location("custom_preprocess", filepath)
    print(f"Loaded pytorch preprocessing function from {filepath} and function name is 'custom_preprocess'")
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    return custom_module.custom_preprocess  # Ensure the function is named 'custom_preprocess' in the file


def load_labels_dict_from_script(script_path):
    spec = importlib.util.spec_from_file_location("label_module", script_path)
    label_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_module)

    if hasattr(label_module, "get_labels_dict"):
        return label_module.get_labels_dict()
    else:
        raise ValueError(f"No function named get_labels_dict() found in {script_path}")

def load_annotations_file_script(script_path):
    spec = importlib.util.spec_from_file_location("label_module", script_path)
    label_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_module)

    if hasattr(label_module, "get_image_ids"):
        return label_module.get_image_ids()
    else:
        raise ValueError(f"No function named get_image_ids() found in {script_path}")


def run_loadgen_test(model_perf, scenario, mode, log_path, energy_logger,sample_size=None):
    """
    Run MLPerf LoadGen test with robust error handling and memory management.
    """
    print(f"\n=== Starting LoadGen Test: {scenario} mode: {mode} ===\n")

    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "Offline": lg.TestScenario.Offline
    }
    mode_map = {
        "AccuracyOnly": lg.TestMode.AccuracyOnly,
        "PerformanceOnly": lg.TestMode.PerformanceOnly
    }
    performance_Only_query_count = random.randint(1, model_perf.dataset_size())
    print(performance_Only_query_count)

    if mode == "AccuracyOnly" and sample_size is not None:
        count = min(model_perf.dataset_size(), sample_size)# up to 10k

    else:
        count = min(model_perf.dataset_size(), performance_Only_query_count)

    settings = lg.TestSettings()
    settings.performance_issue_same_index = True
    settings.min_query_count = count
    settings.max_query_count = count
    settings.min_duration_ms = 0  # optional: remove time constraint
    settings.scenario = scenario_map[scenario]
    settings.mode = mode_map[mode]
    if mode == "AccuracyOnly" and sample_size is not None:
        max_count = min(model_perf.dataset_size(), sample_size) # up to 10k
    else:
        max_count = min(model_perf.dataset_size(), performance_Only_query_count)
    # Create SUT and QSL objects
    sut = None
    qsl = None

    try:
        sut = lg.ConstructSUT(model_perf.issue_queries, model_perf.flush_queries)
        qsl = lg.ConstructQSL(
            max_count,
            max_count,
            model_perf.load_query_samples,
            model_perf.unload_query_samples
        )

        print(f"Running MLPerf Inference for scenario: {scenario} and mode: {mode}...")
        energy_logger.start()
        print("Starting Issue Queries...")
        lg.StartTest(sut, qsl, settings)
        print("Issue Queries Completed")
        energy_data = energy_logger.stop()
        print(f"MLPerf test completed successfully")

    except Exception as e:
        print(f"Error during LoadGen test: {str(e)}")
        traceback.print_exc()
        # Return empty energy data in case of failure
        energy_data = {"total_energy_wh": 0.0}

    finally:
        # Clean up resources
        print("Cleaning up LoadGen resources...")
        try:
            if qsl is not None:
                lg.DestroyQSL(qsl)
            if sut is not None:
                lg.DestroySUT(sut)
        except Exception as e:
            print(f"Error during LoadGen cleanup: {str(e)}")

        # Force garbage collection
        gc.collect()
        print("LoadGen resources cleanup complete")

    # Move MLPerf logs
    try:
        for log_file in [
            "mlperf_log_accuracy.json",
            "mlperf_log_detail.txt",
            "mlperf_log_summary.txt",
            "mlperf_log_trace.json"
        ]:
            src = Path(log_file)
            if src.exists():
                shutil.move(str(src), log_path / log_file)
    except Exception as e:
        print(f"Error moving log files: {str(e)}")

    print(f"\n=== Completed LoadGen Test: {scenario} mode: {mode} ===\n")
    return energy_data


def main():
    global labels_dict
    parser = argparse.ArgumentParser(description="Run MLPerf model evaluation")
    parser.add_argument("--model", type=str, help="Path to the model file or predefined model name.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_type", choices=["pytorch", "onnx", "tensorflow", "huggingface"],
                        help="Specify model type")
    parser.add_argument("--scenario", type=str, default="SingleStream", choices=["SingleStream", "Offline"])
    parser.add_argument("--mode", type=str, default="AccuracyOnly", choices=["AccuracyOnly", "PerformanceOnly"] )
    parser.add_argument("--preprocess_fn_file", type=str, help="File path to custom preprocessing function.")
    parser.add_argument("--task_type", choices=["classification", "detection", "segmentation"], required=True,
                        help="Type of ML task")
    parser.add_argument("--flops", type=int, required=True, help="Model FLOPs (used for EDE scoring)")
    parser.add_argument("--labels_processing_file", type=str, required=True, help="Path to processing labels dictionary")
    parser.add_argument("--energiBridge", type=str, required=True, help="Path to energibridge executable")
    parser.add_argument("--model_architecture", required=True, type=str, default="resnet18",
                        help="Model architecture: resnet18, efficientnet_b0, alexnet, etc.")
    parser.add_argument("--index_map",  type=str,
                        help="Mappings from index to class name. Required for classification tasks for smaller datasets")
    parser.add_argument("--sample_size", type=str, required=True,
                        help="Size of sample we would like to to evaluate")

    args = parser.parse_args()
    if args.sample_size:
        sample_size = int(args.sample_size)

    # Dynamically load the custom preprocessing function if provided
    custom_preprocess_fn = None
    if args.preprocess_fn_file:
        try:
            custom_preprocess_fn = load_custom_preprocess_fn(args.preprocess_fn_file)
        except Exception as e:
            print(f"Error loading custom preprocess function: {e}")
            sys.exit(1)


    if args.model_architecture and args.model is None:
        experiment_name = Path(args.model_architecture).stem
    else:
        experiment_name = Path(args.model).stem

    if args.labels_processing_file:
        labels_dict = load_labels_dict_from_script(args.labels_processing_file)

    if args.index_map:
        with open(args.index_map, "r") as f:
            index_map = json.load(f)
    # Set up logging directories
    try:
        log_base = Path(f"metrics/logs/{experiment_name}")
        log_base.mkdir(parents=True, exist_ok=True)
        (log_base / "codecarbon").mkdir(parents=True, exist_ok=True)
        mlperf_log_path = Path(f"./mlperf_{experiment_name}_log")
        mlperf_log_path.mkdir(exist_ok=True)
    except Exception as e:
        print(f"Error creating log directories: {e}")
        sys.exit(1)

    # Initialize Model Performance Evaluation
    try:
        print("Initializing ModelPerf...")
        model_perf = ModelPerf(args.model, args.model_type, args.dataset, loadgen=lg, task_type=args.task_type,
                               preprocess_fn=custom_preprocess_fn,
                               model_architecture=args.model_architecture)
    except Exception as e:
        print(f"Error initializing ModelPerf: {e}")
        traceback.print_exc()
        sys.exit(1)

    os_name = platform.system().lower()
    print(f"Running on {os_name}")
    # Initialize Unified Energy Logger
    try:
        base = Path(__file__).resolve().parent.parent.parent  # goes from general/ to vision/
        print(f"Base path: {base}")
        rapl_path = base / args.energiBridge
        print(f"RAPL path: {rapl_path}")

        energy_logger = UnifiedLogger(
            experiment_name=experiment_name,
            cc_output_dir=f'./vision/metrics/logs/{experiment_name}/codecarbon',
            eb_output_file=f'./vision/metrics/logs/{experiment_name}/energibridge.csv',
            rapl_power_path=rapl_path
        )
    except Exception as e:
        print(f"Error initializing energy logger: {e}")
        traceback.print_exc()
        sys.exit(1)
    try:
        print("\n=== Running benchmark ===")
        energy = run_loadgen_test(model_perf, args.scenario, args.mode, mlperf_log_path, energy_logger,sample_size)
        print(f"Accuracy energy result: {energy}")
        # Force garbage collection after test
        gc.collect()
    except Exception as e:
        print(f"Error in AccuracyOnly benchmark: {e}")
        traceback.print_exc()
        energy = {"total_energy_wh": 0.0}
    if args.mode=="PerformanceOnly":
        results_path = Path(f"./latency_results_performanceonly_{experiment_name}.json")
        csv_file = results_path.with_name(f"{results_path.stem}_{os_name}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "latency_ms"])
            for i, latency in enumerate(model_perf.latencies):
                writer.writerow([i, latency])

    #Run accuracy evaluation
    elif args.mode=="AccuracyOnly":
        try:
            print("\n=== Running Accuracy Evaluation ===")
            if args.task_type == 'classification':
                if args.index_map:
                 print("[INFO] Using index map for default accuracy evaluation.")
                 accuracy, total_evaluated = evaluate_classification_accuracy_from_dict(
                        model_perf.predictions_log, labels_dict, index_map)
                else:
                    print("[INFO] Using default direct inference (batch mode) for accuracy evaluation.")
                    accuracy, total_evaluated = evaluate_classification_accuracy(
                        model_perf.predictions_log, labels_dict,args.dataset)
            elif args.task_type == 'detection':
                 accuracy, total_evaluated = evaluate_detection_accuracy(model_perf.predictions_log, args.dataset, load_annotations_file_script(args.labels_processing_file))
                 print(f"Evaluated {total_evaluated} samples")
            else:
                print("Segmentation not yet supported.")
                accuracy = 0.0

            print(f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error during accuracy evaluation: {e}")
            traceback.print_exc()
            accuracy = 0.0

        # Calculate EDE Score
        try:
            print("\n=== Running Normalization + EDE Calculation ===")
            normalizer = DualReferenceNormalizer()
            normalized_energy = normalizer.normalize_energy(energy["total_energy_wh"])
            baseline_acc = normalizer.refs['accuracy_threshold'][args.task_type]

            penalty = EDECycleCalculator.compute_penalty(accuracy, baseline_acc)
            ede_score = EDECycleCalculator.compute_ede_cycle(
                accuracy=accuracy,
                flops=args.flops,
                inference_energy=energy["total_energy_wh"],
                alpha=2
            )
            final_ede = ede_score * penalty
        except Exception as e:
            print(f"Error during EDE calculation: {e}")
            traceback.print_exc()
            normalized_energy = 0.0
            penalty = 0.0
            ede_score = 0.0
            final_ede = 0.0

        # Final Reporting
        try:
            print("\n=== Final Benchmark Report ===")
            print(f"Accuracy: {accuracy}")
            print(f"Inference Energy (Wh): {energy['total_energy_wh']}")
            print(f"Normalized Energy: {normalized_energy}")
            print(f"Penalty Factor: {penalty}")
            print(f"Final EDE Score (with penalty): {final_ede}")

            # Write results to file
            results_path = Path(f"./results_{experiment_name}.json")

            os_name = platform.system().lower()
            csv_file = results_path.with_name(f"{results_path.stem}_{os_name}.csv")
            fieldnames = [
                "experiment_name",
                "model_architecture",
                "accuracy",
                "energy_wh",
                "normalized_energy",
                "penalty_factor",
                "ede_score",
                "flops",
                "task_type",
                "timestamp"
            ]
            # Check if file exists to decide whether to write header
            write_header = not os.path.exists(csv_file)

            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "experiment_name": experiment_name,
                    "model_architecture": args.model_architecture,
                    "accuracy": float(accuracy),
                    "energy_wh": float(energy["total_energy_wh"]),
                    "normalized_energy": float(normalized_energy),
                    "penalty_factor": float(penalty),
                    "ede_score": float(final_ede),
                    "flops": args.flops,
                    "task_type": args.task_type,
                    "timestamp": datetime.now().isoformat()
                })

            # with open(results_path, "w") as f:
            #     json.dump(results, f, indent=2)

            print(f"Results saved to {results_path}")

        except Exception as e:
            print(f"Error generating final report: {e}")
            traceback.print_exc()

    print("\n=== Benchmark Complete ===")

    # Final cleanup
    try:
        print("Performing final cleanup...")
        del model_perf
        gc.collect()
        print("Cleanup complete")
    except Exception as e:
        print(f"Error during final cleanup: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error in main: {e}")
        traceback.print_exc()
        sys.exit(1)