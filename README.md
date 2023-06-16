# MOReL Implementation User Guide

This work is our implementation of the paper titled "MOReL : Model-Based Offline Reinforcement Learning", published at NeurIPS 2020.

For detailed explaination of the proposed algoridthms, please refer to the original paper: https://arxiv.org/abs/2005.05951

## Installation

In our work, we use Maze2D and Ant as our learning tasks, and use the dataset from D4RL to train dynamic models.
If you have not had D4RL in your environment, please install D4RL first. We suggest you follow the intructions in this websites to install Mujoco and D4RL:
https://zhuanlan.zhihu.com/p/489475047

### Installation Tips

- Please use ubuntu 18.04 to build up the environment, and you may want to use ubuntu in Linux system or through wsl in Windows.
- If you have had the package gym in your environment, we recommend that you change your gym to version 0.19.0. Because gym of higher version may lead to some errors.
- We ues Pytorch to construct the neural networks in our implementation, so please install `torch` in your environment.

## Quick Start: Train new dynamic models and policy using MOReL from scratch

Please download our work to your local computer and enter one of the folder `MORel-Ant` or `MORel-Maze2D` according to the task you want to test.

### Run
```shell
# start training dynamic models, and then train policy 
python train.py
```
If the training is successfully executed, your terminal will show a progress bar like the picture shown below.
![image](https://github.com/sonicokuo/MOReL/assets/73321093/9c0e97b6-01cb-47e1-99fa-1b79760ad5d7)

Our implementation supports automatically saving checkpoints of models when the training goes on.  
```shell
# If you want to resume the training of dynamic models
python train.py --continue_training --load_epoch_num=30
```

Besides, you can use command-line arguments to manipulate some hyperparameters of training. For detailed information of argument setting, please refer to the file `MOReL-Maze2D/config.py`.

