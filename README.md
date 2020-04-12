This repository contains the PyTorch implementation of the paper Robust Line Segments Matching via Graph
Convolution Networks.

## Requirements
Install pytorch 1.1+
Install ninja-build: 
```setup 
sudo apt-get install ninja-build
```
Install python packages: 
```setup 
pip install tensorboardX scipy easydict pyyaml
```
## Dataset
to train and eval the  network, you should download [Scannet](), and then, you should use the [code]() to pre-process (e.g., generate the grund truth label) the dataset. if you want to augment the dataset, run:


```train
python3 train_eval.py --cfg your_yaml_path
```

## Training

To train the model(s) in the paper, run this command:

```train
python3 train_eval.py --cfg your_yaml_path
```
> 📋Example python3 train_eval.py --cfg

## Evaluation

To evaluate the model on Scannet, run:

```eval
python3 eval.py --cfg your_yaml_path
```
> 📋Example python3 train_eval.py --cfg

## Visualization
To view the matching results, run:

```vis
python3 eval.py --cfg your_yaml_path
```
> 📋the Pre-trained Models will be provided when the paper is accepted.
