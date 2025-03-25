import os
import sys
import argparse
import mlperf_loadgen as lg
from pathlib import Path
from inference import ModelPerf  # Generalized model handler

sys.path.append(os.path.abspath("."))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model file or predefined model name.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--scenario", type=str, default="SingleStream", choices=["SingleStream", "Offline"])
    args = parser.parse_args()

    # Select preprocessing based on model type
    if args.model == "efficientnet":
        from vision.efficientnet.preprocess import preprocess
    elif args.model == "yolo":
        from vision.yolo.preprocess import preprocess
    else:
        from vision.general.preprocess import preprocess  # Generic preprocessing

    model_runner = ModelPerf(args.model, args.dataset, preprocess, loadgen=lg)

    # LoadGen config
    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "Offline": lg.TestScenario.Offline
    }

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.mode = lg.TestMode.PerformanceOnly

    log_path = f"mlperf_{Path(args.model).stem}_log"
    os.makedirs(log_path, exist_ok=True)

    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = log_path
    log_settings.enable_trace = False


    sut = lg.ConstructSUT(
        model_runner.issue_queries,
        model_runner.flush_queries
    )
    qsl = lg.ConstructQSL(
        model_runner.dataset_size(),
        model_runner.dataset_size(),
        model_runner.load_query_samples,
        model_runner.unload_query_samples
    )

    print(f"Running MLPerf Inference for model: {args.model} with dataset: {args.dataset}")
    lg.StartTest(sut, qsl, settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

if __name__ == "__main__":
    main()