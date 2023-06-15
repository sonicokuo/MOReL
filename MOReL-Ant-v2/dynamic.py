import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import os

from dataset import Data

class dynamic(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim, hidden_size=512):
        super(dynamic, self).__init__()

        self.encoder_state = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

        self.encode_action = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

        self.model = nn.Sequential(
            nn.Linear(20, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, state, action):
        s = self.encoder_state(state)
        a = self.encode_action(action)
        x = self.model(torch.cat((s, a), -1))
        return x


class USAD():
    def __init__(self, state_dim, action_dim, output_dim, threshold, opt, model_num=4, device="cuda:0"):
        self.threshold = threshold
        self.device = device
        self.model_num = model_num
        self.opt = opt

        self.models = []

        # Initialize 4 ensemble dynamic models
        for i in range(self.model_num):
            self.models.append(dynamic(state_dim, action_dim, output_dim).to(self.device))

    def forward(self, model_idx, state, action):
        return self.models[model_idx](state, action)

    def train_step(self, idx, state, action, target):
        self.optimizers[idx].zero_grad()

        pred = self.models[idx](state, action)

        loss = self.losses[idx](pred, target)

        loss.backward()

        self.optimizers[idx].step()

        return loss

    def train(self, data, opt, optimizer=torch.optim.Adam, loss=nn.MSELoss, summary_writer=None): #dataloader -> data
        path = "../models/"
        if not os.path.isdir(path):
            os.mkdir(path)

        self.optimizers = [None] * self.model_num
        self.losses = [None] * self.model_num

        # Declare optimizers and losses
        for i in range(self.model_num):
            self.optimizers[i] = optimizer(self.models[i].parameters(), lr=5e-4)
            self.losses[i] = nn.MSELoss()

        # Load pretrained models
        starting_epoch = 0
        if opt.continue_training is True:
            # print('============continue training===========')
            starting_epoch = opt.load_epoch_num + 1  # loading epoch=0 means continuing from epoch=1
            for i in range(self.model_num):
                self.models[i].load_state_dict(torch.load(
                    "../models/dynamic_{train_epoch}_{model_idx}.pt".format(train_epoch=opt.load_epoch_num,
                                                                            model_idx=i)))

        # Create separate dataloaders for each dynamic model
        dataloader_list = [DataLoader(data, batch_size=256, shuffle=True) for i in range(4)]

        # Main train loop
        for epoch in range(starting_epoch, opt.epochs):
            print('epoch={}'.format(epoch))
            for model_idx, dataloader in enumerate(dataloader_list):
                for i, batch in enumerate(tqdm(dataloader)):
                    # target = reward of the real dataset
                    state, action, target = batch

                    loss_val = self.train_step(model_idx, state, action, target)
                    #loss_vals = list(map(lambda i: self.train_step(i, state, action, target), range(self.model_num)))
                    #print(loss_vals)

                    #if epoch % opt.save_freq == 9:
                    torch.save(self.models[model_idx].state_dict(),
                                "../models/dynamic_{train_epoch}_{model_idx}.pt".format(train_epoch=epoch+1, model_idx=model_idx))

                    # Tensorboard
                    #summary_writer.add_scalar("Avg Dynamic Loss", sum(loss_vals) / self.model_num, epoch)
                    summary_writer.add_scalar('Loss/dynamics_{model_idx}'.format(model_idx=model_idx), loss_val, epoch*len(dataloader) + i)

    def checker(self, predictions):
        dis = scipy.spatial.distance_matrix(predictions, predictions)
        return (np.max(dis) > self.threshold)

    def predict(self, state, action):
        with torch.set_grad_enabled(False):
            # For i in 4 dynamic models, predict (i, pred_state, reward), and stack 4 predictions together
            return torch.stack(list(map(lambda i: self.forward(i, state, action), range(self.model_num))))
