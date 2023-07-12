## Dependencies:

Code has been tested on:

* Python 3.6, 3.7
* OS: Windows 10, CentOS 7.9.2009
* GPU: RTX2080Ti, Tesla v100

### Pip
Use the following command to establish the environment:

```shell
pip install -r requirements.txt
```

## Quick start

### Data Source
- We use the original datasets and models from paper: AUTOTRAINER: An Automatic DNN Training
Problem Detection and Repair System. Please download from [here](https://github.com/shiningrain/AUTOTRAINER).
- Download into folder `Evaluation`, sub-folders inside including `MNIST`, `CIFAR-10`, etc.

### Sample of fault diagnosis
There are two original models with their generated mutants under `Evaluation/MNIST`.
Run
```shell
python CodeBook/predict.py
```
to use our pretrained models for fault diagnosis. 
You will see the predicted results of both fault diagnosis models with and without coverage metrics.

##Workflow
Build fault diagnosis models for DNN models on your own.
### Step1 Fault Seeding
```shell
python CodeBook/seed_all.py --base Evaluation --dataset MNIST -sf 1 --fault_type loss
```

The above example seeds a faults in type of `loss` for each DNN model under `Evaluation/MNIST`. 
Specify the objective dataset by `--dataset` and the seeded fault type by `--fault_type`.

### Step2 Mutant runing and runtime data collecting
```shell
python CodeBook/run_all.py --base Evaluation --dataset MNIST --gpu 0 --run 1 --max_iter 1
```
The above example runs each generated mutant once, together with original one under `Evaluation/MNIST`, without using GPU.

### Step3 Coverage Computing
```shell
python CodeBook/compute_coverage.py --base Evaluation --dataset MNIST -iter 1
```
The above example computes the coverage of each generated mutant and original model under `Evaluation/MNIST`

### Step4 Metrics Extraction
```shell
python CodeBook/Utils/cal_stats_1.py -pd=Evaluation -ds=MNIST -iter=5 -ov 1 -stat=1
```
The above example extracts metrics from each case under `Evaluation/MNIST`, which outputs to `Evaluation/MNIST/summary.csv`.

### Step5 Fault Diagnosis Model Construction
```shell
python Codebook/MultiLabelClassification/MultiLabelClassifier.py -pd=evaluation -ds=MNIST -cov=1
```
The above example constructs fault diagnosis models for MNIST based on the features with coverage metrics.
Verify `-cov=0` to construct models without coverage metrics.
The diagnosis models will be under `Classifiers`.


