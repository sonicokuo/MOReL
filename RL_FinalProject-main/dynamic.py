import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial

import numpy as np
from tqdm import tqdm

class dynamic(nn.Module):
	def __init__(self, state_dim, action_dim, output_dim, hidden_size = 512):
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
				nn.Linear(10, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, output_dim)
		)

	def forward(self, state, action):
		s = self.encoder_state(state)
		a = self.encode_action(action)
		x = self.model(s+a)
		return x

class USAD():
	def __init__(self, state_dim, action_dim, output_dim, threshold, model_num = 4, device = "cuda:0"):
		self.threshold = threshold
		self.device = device
		self.model_num = model_num

		self.models = []

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

	def train(self, dataloader, epochs = 5, optimizer = torch.optim.Adam, loss = nn.MSELoss):
		self.optimizers = [None] * self.model_num
		self.losses = [None] * self.model_num

		for i in range(self.model_num):
			self.optimizers[i] = optimizer(self.models[i].parameters())
			self.losses[i] = nn.MSELoss()

		for epoch in range(epochs):
			for i, batch in enumerate(tqdm(dataloader)):
				state, action, target = batch

				loss_val = list(map(lambda i : self.train_step(i, state, action, target), range(self.model_num)))

	def checker(self, predictions):
		dis = scipy.spatial.distance_matrix(predictions, predictions)
		return (np.max(dis) > self.threshold)

	def predict(self, state, action):
		with torch.set_grad_enabled(False):
			return torch.stack(list(map(lambda i : self.forward(i, state, action), range(self.model_num))))