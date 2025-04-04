import json
import os
import sys
import argparse
from time import sleep
import gc
import traceback

import mlperf_loadgen as lg
import shutil
from pathlib import Path

from PIL import Image

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
    print(f"Loaded pytorch preprocessing function from {filepath} and function name is 'custom_preprocess'")
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    return custom_module.custom_preprocess  # Ensure the function is named 'custom_preprocess' in the file
def basic_preprocess(path):
    img = Image.open(path).convert("RGB")
    # We won't do anything fancy – just open & close it:
    img.load()
    return True

# # def run_test(dataset_dir):
# #     files = sorted(os.listdir(dataset_dir))
# #     for i, fname in enumerate(files):
# #         full_path = os.path.join(dataset_dir, fname)
# #         print(f"[TEST] Processing {i}: {full_path}")
# #         try:
# #             basic_preprocess(full_path)
# #             # Optionally do inference if you want:
# #             # tensor = transform(Image.open(full_path))
# #             # output = model(tensor.unsqueeze(0))
# #         except Exception as e:
# #             print(f"[TEST] ERROR on file {fname}: {e}")
# #             break
# #     print("Completed test of 10 images")
# def run_test(dataset_dir, labels_dict_path, model_perf, limit=10):
#     """
#     Runs a small accuracy test on up to 'limit' files in 'dataset_dir'.
#     Uses 'labels_dict_path' to map filenames -> integer labels.
#     """
#     # 1) Load label JSON
#     with open(labels_dict_path, "r") as f:
#         labels_dict = json.load(f)
#
#     # 2) Gather up to 'limit' files
#     files = sorted(os.listdir(dataset_dir))[:limit]
#
#     correct = 0
#     total = 0
#
#     for i, fname in enumerate(files):
#         # We skip non-image files if needed
#         if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue
#
#         # 3) If your labels dict keys are like "0_20551.png", we must match that exact filename
#         if fname not in labels_dict:
#             print(f"[run_test] WARNING: {fname} not found in labels_dict, skipping.")
#             continue
#
#         label = labels_dict[fname]
#
#         # 4) Full path
#         full_path = os.path.join(dataset_dir, fname)
#         print(f"[run_test] Processing {i}: {full_path} - label={label}")
#
#         try:
#             # 5) Preprocess + predict
#             input_tensor = model_perf.preprocess_fn.preprocess(full_path)
#             input_tensor = input_tensor.to(model_perf.device)
#
#             pred = model_perf.predict(input_tensor)
#
#             # Compare
#             if pred == label:
#                 correct += 1
#             total += 1
#
#             print(f"[run_test]  => pred={pred}, label={label}")
#         except Exception as e:
#             print(f"[run_test] ERROR on file {fname}: {e}")
#             break
#
#     # 6) Final accuracy
#     accuracy = (correct / total) if total > 0 else 0.0
#     print(f"[run_test] Completed testing {total} images out of {limit}; accuracy: {accuracy:.4f}")
#     return accuracy
#


def run_loadgen_test(model_perf, scenario, mode, log_path, energy_logger):
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

    count = min(model_perf.dataset_size(), 60000)
    settings = lg.TestSettings()
    settings.performance_issue_same_index = True
    settings.min_query_count = count
    settings.max_query_count = count
    settings.min_duration_ms = 0  # optional: remove time constraint
    settings.scenario = scenario_map[scenario]
    settings.mode = mode_map[mode]
    if mode == "AccuracyOnly":
        max_count = min(model_perf.dataset_size(), 60000)  # up to 10k
    else:
        max_count = min(model_perf.dataset_size(), 500)
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
    parser = argparse.ArgumentParser(description="Run MLPerf model evaluation")
    parser.add_argument("--model", type=str, help="Path to the model file or predefined model name.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_type", choices=["pytorch", "onnx", "tensorflow", "huggingface"],
                        help="Specify model type")
    parser.add_argument("--scenario", type=str, default="SingleStream", choices=["SingleStream", "Offline"])
    parser.add_argument("--preprocess_fn_file", type=str, help="File path to custom preprocessing function.")
    parser.add_argument("--task_type", choices=["classification", "detection", "segmentation"], required=True,
                        help="Type of ML task")
    parser.add_argument("--flops", type=int, required=True, help="Model FLOPs (used for EDE scoring)")
    parser.add_argument("--labels_dict", type=str, required=True, help="Path to labels.json file")
    parser.add_argument("--model_architecture", required=True, type=str, default="resnet18",
                        help="Model architecture: resnet18, efficientnet_b0, alexnet, etc.")

    args = parser.parse_args()

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
        model_perf = ModelPerf(args.model, args.model_type, args.dataset, loadgen=lg,
                               preprocess_fn=custom_preprocess_fn,
                               model_architecture=args.model_architecture)
    except Exception as e:
        print(f"Error initializing ModelPerf: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Load labels dictionary
    try:
        with open(args.labels_dict, "r") as f:
            labels_dict = json.load(f)
    except Exception as e:
        print(f"Error loading labels dictionary: {e}")
        sys.exit(1)

    # Initialize Unified Energy Logger
    try:
        base = Path(__file__).resolve().parent.parent  # goes from general/ to vision/
        print(f"Base path: {base}")
        rapl_path = base / "metrics/EnergiBridge/target/release/energibridge"
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

    #RUN PerformanceOnly Mode
    # try:
    #     print("\n=== Running PerformanceOnly benchmark ===")
    #     perf_energy = run_loadgen_test(model_perf, args.scenario, "PerformanceOnly", mlperf_log_path, energy_logger)
    #     print(f"Performance energy result: {perf_energy}")
    #     # Force garbage collection before next test
    #     gc.collect()
    #     print("Waiting 10 seconds before next test...")
    #     sleep(10)
    # except Exception as e:
    #     print(f"Error in PerformanceOnly benchmark: {e}")
    #     traceback.print_exc()
    #     perf_energy = {"total_energy_wh": 0.0}

    #RUN AccuracyOnly Mode
    try:
        print("\n=== Running AccuracyOnly benchmark ===")
        acc_energy = run_loadgen_test(model_perf, args.scenario, "AccuracyOnly", mlperf_log_path, energy_logger)
        print(f"Accuracy energy result: {acc_energy}")
        # Force garbage collection after test
        gc.collect()
    except Exception as e:
        print(f"Error in AccuracyOnly benchmark: {e}")
        traceback.print_exc()
        acc_energy = {"total_energy_wh": 0.0}

    # test_accuracy = run_test(
    #     dataset_dir=args.dataset,
    #     labels_dict_path="vision/dataset_dir/mnist/mnist_images/labels.json",
    #     model_perf=model_perf,
    #     limit=10  # or however many you want
    # )
    # print(f"[run_test] Test accuracy on {10} files: {test_accuracy:.4f}")

    #Run accuracy evaluation
    try:
        print("\n=== Running Accuracy Evaluation ===")
        #subset = model_perf.get_inferred_indices()
        # print(f"Subset size: {len(subset)}")

        # Use a protective wrapper for the accuracy evaluation
        accuracy = 0.0
        with open("vision/dataset_dir/mnist/labels/labels.json", "r") as f:
            labels_dict = json.load(f)
        if args.task_type == 'classification':
            # predictions=model_perf.infer_all(args.dataset, batch_size=600)
            accuracy, total_evaluated = evaluate_classification_accuracy(model_perf,labels_dict, args.dataset,limit=None)
            print(f"Evaluated {total_evaluated} samples")
        elif args.task_type == 'detection':
            accuracy, total_evaluated = evaluate_detection_accuracy(model_perf, args.dataset, args.labels_dict, subset)
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
        normalized_energy = normalizer.normalize_energy(acc_energy["total_energy_wh"])
        baseline_acc = normalizer.refs['accuracy_threshold'][args.task_type]

        penalty = EDECycleCalculator.compute_penalty(accuracy, baseline_acc)
        ede_score = EDECycleCalculator.compute_ede_cycle(
            accuracy=accuracy,
            flops=args.flops,
            inference_energy=acc_energy["total_energy_wh"],
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
        print(f"Inference Energy (Wh): {acc_energy['total_energy_wh']}")
        print(f"Normalized Energy: {normalized_energy}")
        print(f"Penalty Factor: {penalty}")
        print(f"Final EDE Score (with penalty): {final_ede}")

        # Write results to file
        results_path = Path(f"./results_{experiment_name}.json")
        results = {
            "experiment_name": experiment_name,
            "model_architecture": args.model_architecture,
            "accuracy": float(accuracy),
            "energy_wh": float(acc_energy["total_energy_wh"]),
            "normalized_energy": float(normalized_energy),
            "penalty_factor": float(penalty),
            "ede_score": float(final_ede),
            "flops": args.flops,
            "task_type": args.task_type
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

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