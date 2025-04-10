import csv
import json
import os
import platform
import random
import sys
import argparse
from datetime import datetime

import gc
import traceback

import importlib.util

import mlperf_loadgen as lg
import shutil
from pathlib import Path


from vision.general.inference import ModelPerf  # Generalized model handler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from vision.metrics.loggers_energy.EDE_Cycle import EDECycleCalculator
from vision.metrics.loggers_energy.accuracy_metrics import (evaluate_classification_accuracy,evaluate_detection_accuracy,
                                                            evaluate_keypoint_detection_accuracy,
                                                            evaluate_segmentation_accuracy,
                                                            evaluate_segmentation_miou,
                                                            evaluate_image_retrieval_accuracy,evaluate_image_captioning_bleu,
                                                            evaluate_classification_accuracy_from_dict,evaluate_panoptic_quality)
from vision.metrics.loggers_energy.dual_reference_normalizer import DualReferenceNormalizer
from vision.metrics.loggers_energy.unified_logger import UnifiedLogger


sys.path.append(os.path.abspath("."))


def load_custom_accuracy_fn(filepath):
    spec = importlib.util.spec_from_file_location("custom_metrics", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "evaluate_accuracy"):
        return module.evaluate_accuracy
    else:
        raise ValueError(f"No function named evaluate_accuracy() found in {filepath}")




def load_custom_preprocess_fn(filepath, dataset_dir=None):
    """
    Dynamically loads a custom preprocessing function from a Python file.

    Args:
        filepath (str): Path to the Python file containing the custom preprocessing function.
        dataset_dir (str): Optional dataset root directory to inject.

    Returns:
        callable: The custom preprocessing function, possibly wrapped with dataset context.
    """
    spec = importlib.util.spec_from_file_location("custom_preprocess", filepath)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    if not hasattr(custom_module, "preprocess"):
        raise ValueError(f"No function named 'preprocess' found in {filepath}")

    raw_preprocess = custom_module.preprocess

    if dataset_dir:
        def wrapped_preprocess(image_name_or_path):
            full_path = Path(dataset_dir) / image_name_or_path
            return raw_preprocess(str(full_path))

        print(f"[INFO] Loaded custom preprocessing with dataset context from: {dataset_dir}")
        return wrapped_preprocess

    print(f"[INFO] Loaded custom preprocessing function without dataset context.")
    return raw_preprocess



def load_ground_truth_labels(script_path):
    spec = importlib.util.spec_from_file_location("label_module", script_path)
    label_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_module)

    if hasattr(label_module, "proces_custom_labels"):
        return label_module.proces_custom_labels()
    else:
        raise ValueError(f"No function named proces_custom_labels()  found in {script_path}")


def load_annotations_file_script(script_path, annotation_path=None):
    """
    Dynamically load get_image_ids() from a label script,
    optionally passing an annotation path.

    Args:
        script_path (str): Path to Python file (e.g., getlabels_coco.py)
        annotation_path (str, optional): Path to COCO annotation file

    Returns:
        List of image IDs
    """
    spec = importlib.util.spec_from_file_location("label_module", script_path)
    label_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_module)

    if hasattr(label_module, "get_image_ids"):
        try:
            # Attempt to pass annotation_path if accepted
            return label_module.get_image_ids(annotation_path)
        except TypeError:
            # Fallback to no-arg version if path is not accepted
            print("[WARN] get_image_ids() did not accept annotation_path, calling without it.")
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
    global labels_values, post_processing, onnx_postprocess_fn, custom_annotations_file, tensorflow_postprocess_fn, reference_config
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
    parser.add_argument("--labels_processing_file", type=str, help="Path to processing labels dictionary")
    parser.add_argument("--energiBridge", type=str, required=True, help="Path to energibridge executable")
    parser.add_argument("--model_architecture", required=True, type=str, default="resnet18",
                        help="Model architecture: resnet18, efficientnet_b0, alexnet, etc.")
    parser.add_argument("--index_map",  type=str,
                        help="Mappings from index to class name. Required for classification tasks for smaller datasets")
    parser.add_argument("--sample_size", type=str, required=True,
                        help="Size of sample we would like to to evaluate")
    parser.add_argument("--accuracy_fn", type=str, required=True,
                        help="Accuracy function to use for computer vision tasks.")
    parser.add_argument("--alpha", type=str, required=True,
                        help="Alpha for ede scoring")
    parser.add_argument("--penalty_override", type=str, required=True,
                        help="custom penalty override value")
    parser.add_argument("--labels_dict", type=str, help="Labels JSON for classification.")
    parser.add_argument("--annotations_file", type=str,
                        help="Annotations file for detection/segmentation/keypoints in COCO format.")
    parser.add_argument("--annotations_path", type=str,
                        help="Annotations file for detection/segmentation/keypoints in COCO format.")
    parser.add_argument("--gt_mask_dir", type=str,
                        help="Directory containing semantic segmentation ground truth masks.")
    parser.add_argument("--gt_panoptic_file", type=str, help="Panoptic annotations in COCO format.")
    parser.add_argument("--predictions_file", type=str, help="File for predicted captions (captioning).")
    parser.add_argument("--references_file", type=str, help="File for reference captions (captioning).")
    parser.add_argument("--ground_truth_retrieval", type=str, help="Ground truth mapping for image retrieval.")
    parser.add_argument("--post_process_fn", type=str, help="User-uploaded post-processing function path.")
    parser.add_argument("--reference_config", type=str, help="Reference configuration file for energy calculation")

    args = parser.parse_args()
    if args.reference_config:
        reference_config=args.reference_config
    if args.sample_size:
        sample_size = int(args.sample_size)
    if args.post_process_fn and args.task_type in ["detection", "image_captioning"]:
        post_processing=args.post_process_fn

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
        labels_values = load_ground_truth_labels(args.labels_processing_file)

    custom_accuracy_fn=None
    if args.accuracy_fn:
        custom_accuracy_fn = load_custom_accuracy_fn(args.accuracy_fn)
    if args.annotations_file and args.annotations_path:
        custom_annotations_file = load_annotations_file_script(args.annotations_file,args.annotations_path)

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
        if args.model_type == "onnx":
            onnx_postprocess_fn = post_processing
        else:
            onnx_postprocess_fn = None
        if args.model_type == "tensorflow":
            tensorflow_postprocess_fn = post_processing
        else:
            tensorflow_postprocess_fn = None
        model_perf = ModelPerf(args.model, args.model_type, args.dataset, loadgen=lg, task_type=args.task_type,
                               preprocess_fn=custom_preprocess_fn,
                               model_architecture=args.model_architecture,onnx_postprocess_fn=onnx_postprocess_fn,
                               tensorflow_postprocess_fn=tensorflow_postprocess_fn)
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
    # Run accuracy evaluation
    # Run accuracy evaluation
    elif args.mode == "AccuracyOnly":
        try:
            print("\n=== Running Accuracy Evaluation ===")

            if custom_accuracy_fn:
                accuracy_kwargs = {
                    "model_perf": model_perf,
                    "task_type": args.task_type,
                    "dataset_path": args.dataset,
                }

                # Add optional arguments based on task type
                if args.task_type == "classification":
                    accuracy_kwargs["labels_dict"] = labels_values
                    if args.index_map:
                        accuracy_kwargs["index_map"] = index_map

                elif args.task_type == "detection":
                    accuracy_kwargs["annotations_file"] = custom_annotations_file
                    accuracy_kwargs["annotations_path"] = args.annotations_path

                elif args.task_type == "segmentation":
                    accuracy_kwargs["annotations_file"] = args.annotations_file
                    accuracy_kwargs["gt_mask_dir"] = args.gt_mask_dir
                    accuracy_kwargs["gt_panoptic_file"] = args.gt_panoptic_file

                elif args.task_type == "keypoint_detection":
                    accuracy_kwargs["annotations_file"] = args.annotations_file

                elif args.task_type == "image_captioning":
                    accuracy_kwargs["predictions_file"] = args.predictions_file
                    accuracy_kwargs["references_file"] = args.references_file

                elif args.task_type == "image_retrieval":
                    accuracy_kwargs["ground_truth_file"] = args.ground_truth_retrieval

                # Call the custom function with filtered args
                accuracy, total_evaluated = custom_accuracy_fn(**accuracy_kwargs)

            else:
                if args.task_type == "classification":
                    if args.index_map:
                        accuracy, total_evaluated = evaluate_classification_accuracy_from_dict(
                            model_perf.predictions_log,
                            labels_values,
                            args.index_map
                        )
                    else:
                        accuracy, total_evaluated = evaluate_classification_accuracy_from_dict(
                            model_perf.predictions_log,
                            labels_values
                        )

                elif args.task_type == "detection":
                    image_ids = load_annotations_file_script(args.labels_processing_file)
                    accuracy, total_evaluated = evaluate_detection_accuracy(
                        model_perf.predictions_log,
                        args.dataset,
                        image_ids
                    )

                elif args.task_type == "segmentation":
                    if args.annotations_file:
                        accuracy, total_evaluated = evaluate_segmentation_accuracy(
                            model_perf,
                            args.dataset,
                            args.annotations_file
                        )
                    elif args.gt_mask_dir:
                        accuracy, total_evaluated = evaluate_segmentation_miou(
                            model_perf,
                            args.dataset,
                            args.gt_mask_dir,
                            num_classes=21  # Set appropriately for your dataset
                        )
                    elif args.gt_panoptic_file:
                        accuracy, total_evaluated = evaluate_panoptic_quality(
                            model_perf,
                            args.dataset,
                            args.gt_panoptic_file
                        )
                    else:
                        raise ValueError("No valid segmentation ground truth provided.")

                elif args.task_type == "keypoint_detection":
                    accuracy, total_evaluated = evaluate_keypoint_detection_accuracy(
                        model_perf,
                        args.dataset,
                        args.annotations_file
                    )

                elif args.task_type == "image_captioning":
                    accuracy, total_evaluated = evaluate_image_captioning_bleu(
                        args.predictions_file,
                        args.references_file
                    )

                elif args.task_type == "image_retrieval":
                    accuracy, total_evaluated = evaluate_image_retrieval_accuracy(
                        model_perf,
                        args.dataset,
                        args.dataset,  # Adjust if different directories for query/gallery
                        args.ground_truth_retrieval
                    )

                else:
                    raise ValueError(f"Unsupported task type: {args.task_type}")

            print(f"[INFO] Task: {args.task_type}, Accuracy: {accuracy:.4f}, Evaluated Samples: {total_evaluated}")

        except Exception as e:
            print(f"[ERROR] Accuracy evaluation failed: {e}")
            traceback.print_exc()
            accuracy = 0.0
            total_evaluated = 0

        # Calculate EDE Score
        try:
            args.alpha = float(args.alpha)
            args.penalty_override = float(args.penalty_override)

            print("\n=== Running Normalization + EDE Calculation ===")
            normalizer = DualReferenceNormalizer(reference_config=reference_config)
            normalized_energy = normalizer.normalize_energy(energy["total_energy_wh"])
            baseline_acc = normalizer.refs['accuracy_threshold'][args.task_type]

            penalty = EDECycleCalculator.compute_penalty(accuracy, baseline_acc, args.penalty_override)
            ede_score = EDECycleCalculator.compute_ede_cycle(
                accuracy=accuracy,
                flops=args.flops,
                inference_energy=energy["total_energy_wh"],
                alpha=args.alpha
            )
            final_ede = ede_score * penalty

        except Exception as e:
            print(f"[ERROR] EDE calculation failed: {e}")
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

            write_header = not csv_file.exists()

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

            print(f"[INFO] Results saved to {csv_file}")

        except Exception as e:
            print(f"[ERROR] Generating final report failed: {e}")
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