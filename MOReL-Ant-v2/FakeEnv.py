import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FakeEnv:
	def __init__(self, dynamic_model,
						obs_mean,
						obs_std,
						action_mean,
						action_std,
						delta_std,
						delta_mean,
						reward_mean,
						reward_std,
						start_states,
						start_states_mean,
						start_states_std,
						penalty = -100,
						timeout = 300,
						device = "cuda:0"):

		self.dynamic_model = dynamic_model
		self.penalty = penalty
		self.device = device

		self.obs_mean = torch.tensor(obs_mean).float().to(self.device)
		self.obs_std = torch.tensor(obs_std).float().to(self.device)
		self.action_mean = torch.tensor(action_mean).float().to(self.device)
		self.action_std = torch.tensor(action_std).float().to(self.device)
		self.delta_mean = torch.tensor(delta_mean).float().to(self.device)
		self.delta_std = torch.tensor(delta_std).float().to(self.device)
		self.reward_mean = torch.tensor(reward_mean).float().to(self.device)
		self.reward_std = torch.tensor(reward_std).float().to(self.device)
		self.start_states_mean = torch.tensor(start_states_mean).float().to(self.device)
		self.start_states_std = torch.tensor(start_states_std).float().to(self.device)

		self.start_states = start_states
		self.timeout = timeout

		self.state = None
		self.steps = 0

	def reset(self):
		idx = np.random.choice(self.start_states.shape[0])
		next_state = torch.tensor(self.start_states[idx]).float().to(self.device)
		self.state = (next_state - self.start_states_mean) / self.start_states_std

		return next_state

	def step(self, action):

		action = (action - self.action_mean) / self.action_std

		predictions = self.dynamic_model.predict(self.state, action)

		delta = predictions[:, :-1]
		rewards = predictions[:, -1]

		delta = self.delta_std * torch.mean(delta, 0) + self.delta_mean
		cur_obs = self.obs_std * self.state + self.obs_mean
		next_state = cur_obs + delta

		rewards = self.reward_std * torch.mean(rewards) + self.reward_mean

		self.state = (next_state - self.obs_mean) / self.obs_std

		raven = self.dynamic_model.checker(predictions.cpu().numpy())

		if(out_of_field):
			rewards = self.penalty

		self.steps += 1

		return next_state, rewards, (out_of_field or self.steps > self.timeout)