# This iteration uses the REINFORCE ML algorithm. I followed the tutorial on the Farama Gymnasium tutorial. Their solution was for Mujoco's Inverted Pendulum
# while I used CartPole so I had to make some minor adjustments
# The first 20 episodes avg points: 12.1
# Last 20 episodes avg points: 395.72

from reinfroce import *
import gymnasium as gym
import random
import pandas as pd
import numpy as np
import time

# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1', render_mode='rgb_array')
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

# env = gym.make('InvertedPendulum-v4')
# wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

# print("obs:", env.observation_space.shape[0])
# print("action:", env.action_space.shape)


t0 = time.time()
total_num_episodes = 5000
obs_space_dims = env.observation_space.shape[0]
# action_space_dims = env.action_space
action_space_dims = 1

agent = REINFROCE(obs_space_dims, action_space_dims)
reward_over_episodes = []

for episode in range(total_num_episodes):
  obs, info = wrapped_env.reset()
  # if episode >= 20 or episode < (total_num_episodes - 20):
  #   wrapped_env.close()

  done = False
  while not done:
    action = agent.sample_action(obs)

    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    agent.rewards.append(reward)

    done = terminated or truncated

  reward_over_episodes.append(wrapped_env.return_queue[-1])
  agent.update()

first_20 = [x[0] for x in reward_over_episodes[0:20]]
print("First 20:", first_20)
avg_first_20 = sum(first_20) / 20
print("Average of first 20:", avg_first_20)

last_20 = [x[0] for x in reward_over_episodes[-20:]]
print("Last 20:", last_20)
avg_last_20 = sum(last_20) / 20
print("Average of last 20:", avg_last_20)

print("Total time:", time.time() - t0)