This repository contains the PyTorch implementation of the paper [Robust Line Segments Matching via Graph
Convolution Networks](https://arxiv.org/abs/2004.04993).

## Requirements
install python3.5.2,pytorch 1.1+,ninja-build: 
```setup 
sudo apt-get install ninja-build
```
Install python packages: 
```setup 
pip install tensorboardX scipy easydict pyyaml
```
## Dataset
To train and eval the network, you should download [Scannet](http://www.scan-net.org/), and then, you should use the [code](https://github.com/mameng1/FindMatchedLine) to pre-process (e.g., generate the grund truth label) the dataset. if you want to augment the dataset, install:
```aug_in
pip install imgaug
```
and then, run:
```aug
python3 aug_scannet.py
```

## Training

To train the model(s) in the paper, run this command:

```train
python3 train_eval.py --cfg your_yaml_path
```
> 📋Example python3 train_eval.py --cfg experiments/vgg16_scannet.yaml

## Evaluation

To evaluate the model on Scannet, run:

```eval
python3 eval.py --cfg your_yaml_path
```
> 📋Example python3 eval.py --cfg experiments/vgg16_scannet.yaml

## Visualization
To view the matching results, run:

```vis
python3 test.py --cfg experiments/vgg16_scannet.yaml --model_path params_last.pt --left_img test_data/000800.jpg --right_img test_data/000900.jpg --left_lines test_data/000800.txt --right_lines test_data/000900.txt
```
> 📋the pre-trained model trained on scannet will be provided when the paper is accepted.
A example is:
<center class="half">
    <img src="https://github.com/mameng1/GraphLineMatching/blob/master/test_data/000800.jpg"  width="300" alt="left"/>
</center>
<center class="half">
    <img src=https://github.com/mameng1/GraphLineMatching/blob/master/test_data/000900.jpg width="300" alt="right"/>
</center>
<center class="half">
    <img src=https://github.com/mameng1/GraphLineMatching/blob/master/test_data/res.png  width="600" alt="res"/>
</center>
