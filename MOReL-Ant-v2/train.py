import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from morel import Morel
from dataset import Data

#
import gym
import d4rl
import os
from datetime import datetime

import numpy as np
import argparse
import config


def main():
    # Create directory for results
    log_dir = "../results/"
    if (not os.path.isdir(log_dir)):
        os.mkdir(log_dir)
    opt = config.get_options()
    run_log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(run_log_dir)

    tensorboard_dir = os.path.join(run_log_dir, "tensorboard_record")
    writer = SummaryWriter(tensorboard_dir)

    # print(gym.__version__)
    # print(d4rl.__version__)
    data = Data("ant-expert-v2")
    dataloader = DataLoader(data, batch_size=256, shuffle=True)

    model = Morel(data.state_dim, data.action_dim, writer, opt)

    model.train(data, dataloader)

    # Evaluate the rewards
    model.eval(data.env)


if __name__ == '__main__':
    main()
