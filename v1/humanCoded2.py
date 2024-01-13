# This iteration was to try to make a basic human solution. More importantly learning more about observation.
# This algorithm is trying to simply keep the pole angle at 0 while also not swinging the momentum too much
# Average steps: 500
import gymnasium as gym
import numpy as np
import time

# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1', render_mode='rgb_array')
observation, info = env.reset()

# print(env.observation_space) 

episodes = 100
episodeSteps = 500

totalScore=0

for episodeIndex in range(episodes):
  observaion, info = env.reset()
  env.render()
  currentScore = 0
  for index in range(episodeSteps):
    if observation[2] > 0:
      action=1
    else:
      action=0
    if observation[3] > .6:
      action=1
    elif observation[3] < -.6:
      action=0
    observation, reward, terminated, truncated, info =env.step(action)
    currentScore += reward
    if terminated:
      break
  totalScore += currentScore

print("Average Score:", totalScore/episodes)