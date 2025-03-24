import os
import sys
import argparse
import time
import numpy as np
import mlperf_loadgen as lg
from pathlib import Path

# Make sure imports work
sys.path.append(os.path.abspath("."))

def get_model_runner(model_name):
    if model_name == "efficientnet":
        from efficientnet.inference import EfficientNetMLPerf
        return EfficientNetMLPerf()
    elif model_name == "yolo":
        from yolo.inference import YOLOMLPerf
        return YOLOMLPerf()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["efficientnet", "yolo"])
    parser.add_argument("--scenario", type=str, default="SingleStream", choices=["SingleStream", "Offline"])
    args = parser.parse_args()

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

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    sut = lg.ConstructSUT(
        model_runner.issue_queries,
        model_runner.flush_queries
    )
    qsl = lg.ConstructQSL(
        100,  # total samples
        100,  # active samples
        model_runner.load_query_samples,
        model_runner.unload_query_samples
    )

    print(f"Running MLPerf Inference for model: {args.model}")
    lg.StartTest(sut, qsl, settings, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

if __name__ == "__main__":
    main()
