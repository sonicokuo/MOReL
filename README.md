# MOReL Implementation User Guide

This work is our implementation of the paper titled "MOReL : Model-Based Offline Reinforcement Learning", published at NeurIPS 2020.

For detailed explaination of the proposed algoridthms, please refer to the original paper: https://arxiv.org/abs/2005.05951

## Installation

In our work, we use Maze2D and Ant as our learning tasks, and use the dataset from D4RL to train dynamic models.
If you have not had D4RL in your environment, please install D4RL first. We suggest you follow the intructions in this websites to install Mujoco and D4RL:
https://zhuanlan.zhihu.com/p/489475047

### Installation Tips

If you have had the package gym in your environment, we recommend that you change your gym to version 0.19.0. Because gym of higher version may lead to some errors.

## Quick Start: Train new dynamic models and policy using MOReL from scratch

Please download our work to your local computer and enter one of the folder `MORel-Ant` or `MORel-Maze2D` according to the task you want to test.
