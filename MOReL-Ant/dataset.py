import torch
from torch.utils.data import Dataset

#import minari

import numpy as np

#
import gym
import d4rl

class Data(Dataset):
	def __init__(self, data_name, episodes=1, device="cuda:0"):
		# dataset = minari.load_dataset(data_name)
		# dataset.set_seed(1)

		# dataset = dataset.sample_episodes(n_episodes=episodes)
		# dataset = dataset[0]

		# self.obs = dataset.observations[1:-1]
		# self.action = dataset.actions[1:-1]
		# self.delta = dataset.observations[2:] - self.obs
		# self.reward = dataset.rewards[2:]

  		# self.state_dim = dataset.observations[0].shape[0]
		# self.action_dim = dataset.actions[0].shape[0]

		# self.initial_obs = dataset.observations[0]

		#
		self.env = gym.make(data_name)
		#self.env.reset()
		#self.env.step(self.env.action_space.sample())
		dataset = self.env.get_dataset()

		self.obs = dataset["observations"][1:-1]
		self.action = dataset["actions"][1:-1]
		self.delta = dataset["observations"][2:] - self.obs
		self.reward = dataset["rewards"][2:]

		self.state_dim = dataset["observations"][0].shape[0]
		self.action_dim = dataset["actions"][0].shape[0]

		self.initial_obs = dataset["observations"][0]

		self.obs_mean = self.obs.mean()
		self.obs_std = self.obs.std()

		self.action_mean = self.action.mean()
		self.action_std = self.action.std()

		self.delta_mean = self.delta.mean()
		self.delta_std = self.delta.std()

		self.reward_mean = self.reward.mean()
		self.reward_std = self.reward.std()

		self.obs = (self.obs - self.obs_mean) / self.obs_std
		self.action = (self.action - self.action_mean) / self.action_std
		self.delta = (self.delta - self.delta_mean) / self.delta_std
		self.reward = (self.reward - self.reward_mean) / self.reward_std

		self.device = device

	def __getitem__(self, idx):
		feed1 = torch.FloatTensor(self.obs[idx]).to(self.device)
		feed2 = torch.FloatTensor(self.action[idx]).to(self.device)
		target = torch.FloatTensor(np.concatenate([self.delta[idx], self.reward[idx:idx+1]])).to(self.device)
		return feed1, feed2, target

	def __len__(self):
		return (len(self.obs)-1)








