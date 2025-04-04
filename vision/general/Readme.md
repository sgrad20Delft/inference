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
#### Energibridge
Make sure to have energibridge installed in your local machine, particularly under in vision/metrics of project repo. For installation, refer to the github page of energibridge(https://github.com/tdurieux/EnergiBridge/tree/main)

### Running

To run a test on a model, run main.py in vision/general. Main.py takes a number of arguments:

- model. Either the path to the model file or the name of the model on huggingface.
- dataset. The path to the dataset of images to classify.
- model_type. The type of model to test. Possible choices are ["pytorch", "onnx", "huggingface"].
- scenario. The scenario to run the test in. Possible choices are ["SingleStream", "Offline"].
- preprocess_fn_file. Path to a file with a custom preprocessing function (optional)
- flops. Number of floating point operations 
- task_type. Type of computer vision task
- model-architecture. Type of model to be installed

Example:

```console
python main.py --model=google/vit-base-patch16-224-in21k --data=D:\\Program Files (x86)\\inference\\mnist_images",
              --scenario=SingleStream --model_type=huggingface
```
Current readme:
```console
python3 vision/general/main.py \                             
  --model google/vit-base-patch16-224-in21k
  --model_type huggingface \
  --model_architecture vit-base-patch16-224-in21k\
  --dataset vision/dataset_dir/mnist/mnist_images \
  --scenario Offline \
  --task_type classification \
  --labels_dict vision/dataset_dir/mnist/labels.json \
  --flops 390000000 \
  --preprocess_fn_file vision/general/preprocess.py
```