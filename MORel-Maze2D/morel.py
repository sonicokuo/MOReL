import torch
import torch.nn
import torch.nn.functional as F

import numpy as np

from dynamic import USAD
from FakeEnv import FakeEnv
from dataset import Data
from PPO import PPO2
from tqdm import tqdm

class Morel:
	def __init__(self, state_dim, action_dim, writer, opt, device = "cuda:0"):
		self.opt = opt
		self.dynamic = USAD(state_dim, action_dim, state_dim+1, 1.0, opt)
		self.policy = PPO2(state_dim, action_dim, opt)
		self.device = device
		self.writer = writer

	def train(self, data, dataloader):
		self.data = data
		self.dataloader = dataloader

		print("---------------- Beginning Dynamics Training ----------------")
		# Train dynamic model
		self.dynamic.train(self.data, self.opt, summary_writer=self.writer) #self.dataloader -> self.data
		print("---------------- Ending Dynamics Training ----------------")

		# Use the trained dyncamic model to build a fake environment
		fake_env = FakeEnv(self.dynamic,
							self.data.obs_mean,
							self.data.obs_std,
							self.data.action_mean,
							self.data.action_std,
							self.data.delta_mean,
							self.data.delta_std,
							self.data.reward_mean,
							self.data.reward_std,
							self.data.initial_obs,
							self.data.initial_obs_mean,
							self.data.initial_obs_std)

		print("---------------- Beginning Policy Training ----------------")
		# Train PPO
		self.policy.train(fake_env, summary_writer=self.writer)
		print("---------------- Ending Policy Training ----------------")

	# Evaluate PPO model average rewards of every episode (given the # of trajectories), and the final evaluation reward
	def eval(self, env):
		print("---------------- Beginning Policy Evaluation ----------------")
		total_rewards = []
		# Evaluate PPO using a # of trajectories
		num_trajectories = 50
		for i in tqdm(range(num_trajectories)):
			# Experience replay with a # of trajectory steps
			num_trajectory_steps = 512
			_, _, _, _, _, _, _, info = self.policy.generate_experience(env, num_trajectory_steps, 0.95, 0.99)
			total_rewards.extend(info["episode_rewards"])

			if self.writer is not None:
				self.writer.add_scalar('Metrics/eval_episode_reward', sum(info["episode_rewards"])/len(info["episode_rewards"]), i)

		print("Final evaluation reward: {}".format(sum(total_rewards)/len(total_rewards)))

		print("---------------- Ending Policy Evaluation ----------------")

