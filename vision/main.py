import os
import sys
import argparse
import numpy as np
from datasets.mnist_loader import prepare_mnist_images
import mlperf_loadgen as lg
from pathlib import Path

# Make sure imports work
sys.path.append(os.path.abspath("."))

def get_model_runner(model_name):
    if model_name == "efficientnet":
        from efficientnet.inference import EfficientNetMLPerf
        return EfficientNetMLPerf(lg)
    elif model_name == "yolo":
        from yolo.inference import YOLOMLPerf
        return YOLOMLPerf(lg)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "yolo"])
    parser.add_argument("--scenario", type=str, default="SingleStream", choices=["SingleStream", "Offline"])
    args = parser.parse_args()

    # Generate and store MNIST images
    mnist_images = prepare_mnist_images()

    model_runner = get_model_runner(args.model)

    # LoadGen config
    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "Offline": lg.TestScenario.Offline
    }

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.mode = lg.TestMode.PerformanceOnly

    log_path = f"mlperf_{args.model}_log"
    os.makedirs(log_path, exist_ok=True)

    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = log_path
    log_settings.enable_trace = False

    sut = lg.ConstructSUT(
        model_runner.issue_queries,
        model_runner.flush_queries
    )
    qsl = lg.ConstructQSL(
        len(mnist_images),  # Total MNIST samples
        len(mnist_images),  # Active samples
        model_runner.load_query_samples,
        model_runner.unload_query_samples
    )

    print(f"Running MLPerf Inference for model: {args.model}")
    lg.StartTest(sut, qsl, settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

if __name__ == "__main__":
    main()