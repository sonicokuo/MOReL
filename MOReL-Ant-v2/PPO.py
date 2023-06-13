import torch
import torch.nn as nn
import torch.nn.functional as F
from FakeEnv import FakeEnv

import numpy as np
from tqdm import tqdm
import os

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim,
                output_dim,
                n_neurons = 64,
                activation = nn.Tanh,
                distribution = torch.distributions.multivariate_normal.MultivariateNormal):

        super(ActorCriticPolicy, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.distribution = distribution

        # Policy Network
        self.h0 = nn.Linear(input_dim, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)

        # Value Network
        self.h0v = nn.Linear(input_dim, n_neurons)
        self.h0_actv = activation()
        self.h1v = nn.Linear(n_neurons, n_neurons)
        self.h1_actv = activation()
        self.value_head = nn.Linear(n_neurons, 1)

        self.var = torch.nn.Parameter(torch.tensor([0.0] * output_dim).cuda(), requires_grad = True)

        self.mean_activation = nn.Tanh()

    def forward(self, obs, action = None):
        # Policy Forward Pass
        x = self.h0(obs)
        x = self.h0_act(x)
        x = self.h1(x)
        x = self.h1_act(x)
        action_logit = self.output_layer(x)

        # Generate action distribution
        mean = action_logit[:,0:self.output_dim]
        var = torch.exp(self.var)
        action_dist = self.distribution(mean, torch.diag_embed(var))

        # Sample action if not passed as argument to function
        # Action is passed when doing policy updates
        if action is None:
            action = action_dist.sample()
            #action = torch.tanh(action)

        neg_log_prob = action_dist.log_prob(action) * -1.
        entropy = action_dist.entropy()

        # Value Forward Pass
        x = self.h0v(obs)
        x = self.h0_actv(x)
        x = self.h1v(x)
        x = self.h1_actv(x)
        value = self.value_head(x)
        value = torch.squeeze(value)

        return action, neg_log_prob, entropy, value


class PPO2():
    def __init__(self, input_dim, output_dim, opt, device = "cuda:0", network = ActorCriticPolicy):

        # Store device on which to allocate tensors
        self.device = device

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Instantiate Actor Critic Policy
        self.policy = network(input_dim, output_dim).to(self.device)
        self.opt = opt

    def forward(self, observation, action = None):
        return self.policy(observation, action = action)

    def generate_experience(self, env, n_steps, gamma, lam):

        # Reset environment on first step
        done = True

        # Initialize memory buffer
        mb_obs, mb_rewards, mb_actions, mb_values, mb_done, mb_neg_log_prob = [],[],[],[],[],[]
        info = {
            "episode_rewards" : [],
            "HALT" : 0
        }
        rewards = 0
        total_reward = 0
        env_info = {}

        # For n in range number of steps
        with torch.set_grad_enabled(False):
            for i in range(n_steps):
                if(done):

                    info["HALT"] += env_info.get("HALT", 0)

                    obs = env.reset()
                    done = False

                    # Convert obs to torch tensor
                    if(not isinstance(env, FakeEnv)):
                        obs = torch.from_numpy(obs.copy()).float().to(self.device)
                    obs = torch.unsqueeze(obs, 0)

                    info["episode_rewards"].append(total_reward)
                    total_reward = 0

                # Choose action
                action, neg_log_prob, _, value = self.forward(observation = obs)

                # Retrieve values
                action = torch.squeeze(action)
                neg_log_prob = torch.squeeze(neg_log_prob)
                value = torch.squeeze(value)

                # Append data from step to memory buffer
                # mb_obs.append(obs.copy())
                mb_obs.append(obs)
                mb_actions.append(action)
                mb_values.append(value)
                mb_neg_log_prob.append(neg_log_prob)
                mb_done.append(done)

                # Step the environment, get new observation and reward
                # If we are interacting with a FakeEnv, we can safely keep the action as a torch tensor
                # Else, we must convert to a numpy array
                # If obs comes as numpy array, convert to torch tensor as well
                if(isinstance(env, FakeEnv)):
                    obs, rewards, done = env.step(action)

                    total_reward += rewards.cpu().item()

                    #rewards = torch.tensor(rewards).float().to(self.device)
                else:
                    obs, rewards, done, env_info = env.step(action.cpu().numpy())

                    obs = torch.from_numpy(obs.copy()).float().to(self.device)

                    total_reward += rewards

                    rewards = torch.tensor(rewards).float().to(self.device)

                # Append reward to memory buffer as well
                mb_rewards.append(rewards)

                obs = torch.unsqueeze(obs, 0)
                if(self.render):
                    env.render()

            # Convert memory buffer lists to numpy arrays
            # print(mb_obs[0:5])
            mb_obs = torch.cat(mb_obs, 0)
            mb_rewards = torch.stack(mb_rewards)
            mb_actions = torch.stack(mb_actions)
            mb_values = torch.stack(mb_values)
            mb_neg_log_prob = torch.stack(mb_neg_log_prob)
            mb_done = np.asarray(mb_done, dtype=bool)

            # Get value function for last state
            _, _, _, last_value = self.forward(obs)

            # Compute generalized advantage estimate by bootstrapping
            mb_advs = torch.zeros_like(mb_rewards).float().to(self.device)
            last_gae_lam = torch.Tensor([0.0]).float().to(self.device)
            for t in reversed(range(n_steps)):
                # next_non_terminal stores index of the next time step (in reverse order) that is non-terminal
                if t == n_steps - 1:
                    # 1 if last step was non-terminal
                    # 0 if last step was terminal
                    next_non_terminal = 1.0 - done
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - mb_done[t+1]
                    next_values = mb_values[t+1]

                delta = mb_rewards[t] + gamma * next_values * next_non_terminal - mb_values[t]
                mb_advs[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam

            # Compute value functions
            mb_returns = mb_advs + mb_values

        return mb_rewards, mb_obs, mb_returns, mb_done, mb_actions, mb_values, mb_neg_log_prob, info

    def train_step(self, clip_range,
                        entropy_coef,
                        value_coef,
                        obs,
                        returns,
                        dones,
                        old_actions,
                        old_values,
                        old_neg_log_probs):

        # Calculate and normalize the advantages
        with torch.set_grad_enabled(False):
            advantages = returns - old_values

            # Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Set policy network to train mode
        self.policy.train()
        with torch.set_grad_enabled(True):
            # Feed batch through policy network
            actions, neg_log_probs, entropies, values = self.forward(obs, action = old_actions)

            loss, pg_loss, value_loss, entropy_mean, approx_kl = self.loss(clip_range,
                                                                            entropy_coef,
                                                                            value_coef,
                                                                            returns,
                                                                            values,
                                                                            neg_log_probs,
                                                                            entropies,
                                                                            advantages,
                                                                            old_values,
                                                                            old_neg_log_probs)

            # Backprop from loss
            loss.backward()

            return loss, pg_loss, value_loss, entropy_mean, approx_kl


    def train(self, env, optimizer = torch.optim.Adam,
                        lr =  0.00027,
                        n_steps = 512,
                        time_steps = 512000,
                        clip_range = 0.2,
                        entropy_coef = 0.01,
                        value_coef = 0.5,
                        num_batches = 5,
                        gamma = 0.99,
                        lam = 0.95,
                        max_grad_norm = 0.5,
                        num_train_epochs = 4,
                        summary_writer = None,
                        render = False):

        self.render = render

        # Total number of train cycles to run
        n_updates = int(time_steps // n_steps)

        # Instantiate optimizer
        self.policy_optim = optimizer(self.policy.parameters(), lr = lr)

        starting_epoch = 0
        # load policy
        if self.opt.continue_training_policy is True:
            starting_epoch = self.opt.load_epoch_num_policy + 1
            self.policy.load_state_dict(torch.load(
                "../models/policy_{train_epoch}.pt".format(train_epoch=self.opt.load_epoch_num_policy)))

        # main train loop
        # n_updates = 5
        for update in tqdm(range(starting_epoch, n_updates)):
            # Collect new experiences using the current policy
            rewards, obs, returns, dones, actions, values, neg_log_probs, info = self.generate_experience(env, n_steps, gamma, lam)
            indices = np.arange(n_steps)

            # Loop over train epochs
            for i in range(num_train_epochs):
                # Shuffle order of data
                np.random.shuffle(indices)

                # Calculate size of each batch
                batch_size = n_steps // num_batches

                # If not evenly divisible, add 1 sample to each batch, last batch will automatically be smaller
                if(n_steps % num_batches):
                    batch_size +=1

                # Loop over batches in single epoch
                for batch_num in range(num_batches):
                    # Reset gradients
                    self.policy.zero_grad()

                    # Get indices for batch
                    if(batch_num != num_batches - 1):
                        batch_indices = indices[batch_num*batch_size:(batch_num + 1)*batch_size]
                    else:
                        batch_indices = indices[batch_num*batch_size:]

                    # Generate batch
                    batch = (arr[batch_indices] for arr in (obs, returns, dones, actions, values, neg_log_probs))

                    # Run train step on batch
                    loss, pg_loss, value_loss, entropy, approx_kl = self.train_step(clip_range, entropy_coef, value_coef, *batch)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)

                    # Run optimizer step
                    self.policy_optim.step()

            if update % self.opt.save_freq == 0:
                torch.save(self.policy.state_dict(), "../models/policy_{train_epoch}.pt".format(train_epoch=update))

            # Tensorboard
            if(summary_writer is not None):
                summary_writer.add_scalar('Loss/total', loss, update*n_steps)
                summary_writer.add_scalar('Loss/policy', pg_loss, update*n_steps)
                summary_writer.add_scalar('Loss/value', value_loss, update*n_steps)
                summary_writer.add_scalar('Metrics/entropy', entropy, update*n_steps)
                summary_writer.add_scalar('Metrics/approx_kl', approx_kl, update*n_steps)
                summary_writer.add_scalar('Metrics/max_reward', sum(info["episode_rewards"])/len(info["episode_rewards"]), update*n_steps)


    def loss(self, clip_range,
                    entropy_coef,
                    value_coef,
                    returns,
                    values,
                    neg_log_probs,
                    entropies,
                    advantages,
                    old_values,
                    old_neg_log_probs):

        ## Entropy loss ##
        entropy_loss = entropies.mean()

        ## Value Loss ##
        # Clip value update
        values_clipped = old_values + torch.clamp(values - old_values, min=-clip_range, max=clip_range)

        value_loss1 = (values - returns)**2
        value_loss2 = (values_clipped - returns)**2

        value_loss = .5 * torch.mean(torch.max(value_loss1, value_loss2))

        ## Policy loss ##
        ratios = torch.exp(old_neg_log_probs - neg_log_probs)

        pg_losses1 = -advantages * ratios
        pg_losses2 = -advantages * torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)

        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

        approx_kl = 0.5 * torch.mean((neg_log_probs - old_neg_log_probs)**2)

        ## Total Loss ##
        loss = pg_loss - (entropy_loss * entropy_coef) + (value_loss * value_coef)

        return loss, pg_loss, value_loss, entropy_loss, approx_kl

    def save(self, save_dir):
        torch.save(self.policy.state_dict(), os.path.join(save_dir, "policy.pt"))

    def load(self, load_dir):
        self.policy.load_state_dict(torch.load(os.path.join(load_dir, "policy.pt")))

    def eval(self, obs):

        # For n in range number of steps
        with torch.set_grad_enabled(False):
            # Convert obs to torch tensor
            # if(not isinstance(env, FakeEnv)):
            obs = torch.from_numpy(obs.copy()).float().to(self.device)
            obs = torch.unsqueeze(obs, 0)

            # Choose action
            action, _, _, _ = self.forward(observation = obs)

        return torch.squeeze(action)
