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
- We use Pytorch to construct the neural networks in our implementation, so please install `torch` in your environment.

## Quick Start: Train new dynamic models and policy using MOReL from scratch

Please download our work to your local computer and enter one of the folder `MORel-Ant` or `MORel-Maze2D` according to the task you want to test.

### Run
```shell
# start training dynamic models, and then train policy 
python train.py
```

As the training goes on, a folder `results` will be created to save tensorboard records. and a folder `models` will be created to save model checkpoints.

If the training is successfully executed, your terminal will show a progress bar like the picture shown below.
![begindynamic](https://github.com/sonicokuo/MOReL/assets/73321093/7ef96769-e8a6-495a-8274-c30664025b6d)
Our implementation supports automatically saving checkpoints of models when the training goes on.  
```shell
# If you want to resume the training of dynamic models
python train.py --continue_training --load_epoch_num=30
```
If the training of dynamic models is accomplished, the training of policy begins, and your terminal will show a progress bar like the picture shown below.
![beginpolicy](https://github.com/sonicokuo/MOReL/assets/73321093/48044214-abc6-4158-9e15-5ec7d893fb5f)

Besides, you can use command-line arguments to manipulate some hyperparameters of training. For detailed information of argument setting, please refer to the file `MOReL-Maze2D/config.py`.

### Issues and future works
For the task Maze2D, we have successfully train a good policy with our implementation. As for the task Ant, we found that the final policy does not converge to a good one. In this stage, we inference that it is caused by the tuning problem of PPO, and the dynamic model seems to be correct. We will further investigate and tackle this issue in the future. If you have any suggestions, please feel free to share with us!

## Acknowledgements

We reference the following repositories to accomplish our implementation:
- https://github.com/SwapnilPande/MOReL
- https://github.com/nikhilbarhate99/PPO-PyTorch


