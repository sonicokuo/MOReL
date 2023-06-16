# MOReL Implementation User Guide

This document provides a comprehensive user guide for our MOReL (Model-Based Offline Reinforcement Learning) implementation. Our work is based on the research paper titled "MOReL: Model-Based Offline Reinforcement Learning," published at NeurIPS 2020. For a detailed explanation of the proposed algorithms, please refer to the original paper available at: https://arxiv.org/abs/2005.05951

## Installation

Our implementation utilizes Maze2D and Ant as the learning tasks and relies on the dataset from D4RL to train dynamic models. If you don't have D4RL installed in your environment, please install it first. We recommend following the instructions provided on the following website to install Mujoco and D4RL: https://zhuanlan.zhihu.com/p/489475047

### Installation Tips

- We recommend using Ubuntu 18.04 to set up the environment. You can use either Ubuntu as your operating system or WSL (Windows Subsystem for Linux) in Windows.
- If you already have the `gym` package installed, we suggest downgrading it to version 0.19.0, as higher versions may lead to certain errors.
- Our implementation relies on PyTorch for constructing neural networks, so please ensure that you have `torch` installed in your environment.

## Quick Start: Training New Dynamic Models and Policies using MOReL from Scratch

To get started, download our implementation to your local computer and navigate to either the `MOReL-Ant` or `MOReL-Maze2D` folder based on the task you want to test.

### Run the Training
```shell
# To start training dynamic models and subsequently train the policy:
python train.py
```

During the training process, a `results` folder will be created to store tensorboard records, and a `models` folder will be created to save model checkpoints.

If the training process is executed successfully, your terminal will display a progress bar similar to the image shown below:
![begindynamic](https://github.com/sonicokuo/MOReL/assets/73321093/7ef96769-e8a6-495a-8274-c30664025b6d)
Our implementation automatically saves model checkpoints as the training progresses. If you wish to resume training the dynamic models at a specific epoch, use the following command:
```shell
# For example, resume the checkpoint of 30th epoch
python train.py --continue_training --load_epoch_num=30
```
Once the training of dynamic models is completed, the training of the policy will commence. Your terminal will display a progress bar similar to the image shown below:
![beginpolicy](https://github.com/sonicokuo/MOReL/assets/73321093/48044214-abc6-4158-9e15-5ec7d893fb5f)

Additionally, you can utilize command-line arguments to adjust certain hyperparameters of the training process. For more detailed information regarding argument settings, please refer to the `MOReL-Maze2D/config.py` file.

### Issues and future works
Regarding the Maze2D task, we have successfully trained a high-performing policy using our implementation. However, for the Ant task, we have observed that the final policy does not converge to an optimal solution. At this stage, we believe that this issue may be attributed to a tuning problem in PPO, as the dynamic model appears to be functioning correctly. We plan to conduct further investigations and address this issue in our future work. If you have any suggestions, please feel free to share them with us!

## Acknowledgements

We reference the following repositories to accomplish our implementation:
- https://github.com/SwapnilPande/MOReL
- https://github.com/nikhilbarhate99/PPO-PyTorch


