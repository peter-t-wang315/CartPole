from __future__ import annotations
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from policyNetwork import *

class REINFROCE:
  def __init__(self, obs_space_dims, action_space_dims):
    self.learning_rate = .0001 # Learning rate for policy optimization
    self.gamma = .99 # Discount factor
    self.eps = .000001 # Small number for mathematical stability

    self.probs = [] # Stores probability values of the sampled action
    self.rewards = [] # Stores the corresponding rewards

    self.net = PolicyNetwork(obs_space_dims, action_space_dims)
    self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

  def sample_action(self, state):
    state = torch.tensor(np.array([state]))
    action_means, action_stddevs = self.net(state)

    distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
    action = distrib.sample()
    prob = distrib.log_prob(action)

    # breakpoint()
    action = torch.where(action > 0.5, torch.tensor(1), torch.tensor(0))

    self.probs.append(prob)

    return action.item()

  def update(self):
    running_g = 0
    gs = []

    for R in self.rewards[::-1]:
      running_g = R + self.gamma * running_g
      gs.insert(0, running_g)

    deltas = torch.tensor(gs)

    loss = 0

    for log_prob, delta in zip(self.probs, deltas):
      loss += log_prob.mean() * delta * (-1)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.probs = []
    self.rewards = []



