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


```aug
python3 aug_scannet.py
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
A example is:


<img src="https://github.com/mameng1/GraphLineMatching/blob/master/test_data/000800.jpg" width = "200" height = "300" alt="left" align=center />

![left:](https://github.com/mameng1/GraphLineMatching/blob/master/test_data/000800.jpg) ![right:](https://github.com/mameng1/GraphLineMatching/blob/master/test_data/000900.jpg) 
![result:](https://github.com/mameng1/GraphLineMatching/blob/master/test_data/res.jpg)
