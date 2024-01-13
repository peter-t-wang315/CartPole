# This iteration was to try to make a basic human solution. More importantly learning more about observation.
# This algorithm is trying to simply keep the pole angle at 0
# Average steps: 41.721
import gymnasium as gym
import numpy as np
import time

# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1', render_mode='rgb_array')
observation, info = env.reset()

# print(env.observation_space) 

episodes = 1000
episodeSteps = 100

averageScore=0

for episodeIndex in range(episodes):
  observaion, info = env.reset()
  env.render()
  for index in range(episodeSteps):
    if observation[2] > 0:
      action=1
    else:
      action=0
    observation, reward, terminated, truncated, info =env.step(action)
    if (terminated):
      print(observation)
      averageScore+=index
      # time.sleep(2)
      break

print("Average Score:", averageScore/episodes)