
# Vision Inference Benchmarking Framework

This repository extends MLCommons' inference framework with support for customizable vision models, preprocessing, postprocessing, accuracy evaluation, and energy metrics.

---

## ️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sgrad20Delft/inference.git
cd inference
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3. Install MLCommons LoadGen (C++ wrapper)

For macOS/Linux:

```bash
export CFLAGS="-std=c++14 -O3"
git clone --recursive https://github.com/mlcommons/inference.git /tmp/loadgen
cd /tmp/loadgen/loadgen
python -m pip install .
```

### 4. Install This Project

From the project root:

```bash
pip install -e .
pip install -r vision/general/requirements.txt
```

---

##  EnergiBridge Setup

Make sure EnergiBridge is built and available under:

```
vision/metrics/EnergiBridge/target/release/energibridge
```

Build instructions:  
➡️ [EnergiBridge GitHub](https://github.com/tdurieux/EnergiBridge/tree/main)

---

##  Running Inference

After installation, run your test using the `vision-infer` command:

```bash
vision-infer \
  --model vision/general/models/onnx_models/yolov4.onnx \
  --model_type onnx \
  --model_architecture yolov4 \
  --dataset vision/dataset_dir/coco/coco_images \
  --scenario Offline \
  --mode AccuracyOnly \
  --task_type detection \
  --flops 390000000 \
  --sample_size 10 \
  --preprocess_fn_file vision/general/datasetpreprocess/custom_preprocess.py \
  --annotations_file vision/general/models/labels/cocoimage/getlabels_coco.py \
  --annotations_path vision/dataset_dir/coco/annotations/instances_val2017.json \
  --accuracy_fn vision/metrics/loggers_energy/custom_metrics.py \
  --alpha 0.5 \
  --penalty_override 1.0 \
  --energiBridge vision/metrics/EnergiBridge/target/release/energibridge \
  --post_process_fn vision/general/datapostprocess/postprocess_coco.py \
  --reference_config vision/metrics/loggers_energy/reference_config.json
```

---

## 🧩 Command-Line Arguments

| Argument                  | Description |
|---------------------------|-------------|
| `--model`                 | Path to the model file (ONNX or HuggingFace) |
| `--model_type`            | One of: `onnx`, `pytorch`, `huggingface` |
| `--model_architecture`    | Architecture name (e.g., `yolov4`, `resnet18`) |
| `--dataset`               | Path to input dataset directory |
| `--scenario`              | MLPerf scenario: `Offline`, `SingleStream` |
| `--mode`                  | `AccuracyOnly` or `PerformanceOnly` |
| `--task_type`             | Vision task: `classification`, `detection`, etc. |
| `--flops`                 | Model FLOPs for EDE score calculation |
| `--sample_size`           | Number of samples to evaluate |
| `--preprocess_fn_file`    | Path to a Python file with a custom `preprocess()` function |
| `--annotations_file`      | Script to return image ID mapping from annotations |
| `--annotations_path`      | COCO JSON annotation file |
| `--accuracy_fn`           | Script implementing `evaluate_accuracy()` |
| `--alpha`                 | Alpha hyperparameter for EDE |
| `--penalty_override`      | Custom penalty override value |
| `--energiBridge`          | Path to compiled EnergiBridge binary |
| `--post_process_fn`       | Python file with `postprocess_onnx_output()` |
| `--reference_config`      | JSON config for energy normalization |

---

## 📦 Development Tips

- Use `pip install -e .` to auto-reload changes during development.
- Always activate the virtualenv before running:  
  `source .venv/bin/activate`
- Add `.venv/` to your `.gitignore`.

---

## 🧪 Testing CLI

```bash
vision-infer --help
```

---

## Troubleshooting

- If `vision-infer` isn't recognized after install, re-run:  
  `pip install -e .`
- Make sure EnergiBridge is compiled and executable.


---

##  License

This project builds on MLCommons Inference and includes modifications under the [Apache 2.0 License](LICENSE).

### Model Download 
For trying out different models, use this link  [https://github.com/onnx/models/tree/main](Model Download)