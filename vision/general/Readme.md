### Installation

Navigate to the loadgen folder and run:

```console
$env:CFLAGS = "-std=c++14 -O3"
python -m pip install .
```
For macOS, 
```console
export CFLAGS="-std=c++14 -O3"
python -m pip install .
```

This installs the c++ library version of mlperf.

Then navigate to vision/general and install the python packages from requirements.txt

```console
pip install -r requirements.txt
```

### Running

To run a test on a model, run main.py in vision/general. Main.py takes a number of arguments:

- model. Either the path to the model file or the name of the model on huggingface.
- dataset. The path to the dataset of images to classify.
- model_type. The type of model to test. Possible choices are ["pytorch", "onnx", "huggingface"].
- scenario. The scenario to run the test in. Possible choices are ["SingleStream", "Offline"].
- preprocess_fn_file. Path to a file with a custom preprocessing function (optional)

Example:

```console
python main.py --model=google/vit-base-patch16-224-in21k --data=D:\\Program Files (x86)\\inference\\mnist_images",
              --scenario=SingleStream --model_type=huggingface
```
Current readme:
```console
python3 vision/general/main.py \                             
  --model vision/general/models/efficientnet_b0.pth \
  --model_type pytorch \
  --model_architecture efficientnet_b0 \
  --dataset vision/dataset_dir/mnist/mnist_images \
  --scenario SingleStream \
  --task_type classification \
  --labels_dict vision/dataset_dir/mnist/labels.json \
  --flops 390000000 \
  --preprocess_fn_file vision/general/preprocess.py
```