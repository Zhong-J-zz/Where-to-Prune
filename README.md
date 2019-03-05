# Where-to-Prune
Using LSTM to guide filter-level pruning. LSTM is employed  as an evaluation metric to
generate the pruning decision for each conv layer.
## Installation
To run this script, you need Pytorch and CUDA. This code is written in Pytorch 3.5.
## Running
To run the script with default parameters, 
```python
python train.py
```
`model0.pkl` is a baseline model of VGG19 on CIFAR-10 dataset. This script will prune `model0.pkl` and generate a slimmer model.
