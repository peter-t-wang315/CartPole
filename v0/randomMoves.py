# This iteration was to learn how to use gymnasium and my very first try with anything ai!
# Average steps: 22.2726
import gymnasium as gym
import numpy as np
import time

# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1', render_mode='rgb_array')
observation, info = env.reset()

# print(env.observation_space) 

episodes = 10000
episodeSteps = 100

totalScore = 0

for episodeIndex in range(episodes):
  observaion, info = env.reset()
  env.render()
  currentScore = 0
  for index in range(episodeSteps):
    random_action=env.action_space.sample()
    observation, reward, terminated, truncated, info =env.step(random_action)
    currentScore += reward
    if (terminated):
      break
  totalScore += currentScore

print("Average Score:", totalScore/episodes)