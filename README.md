# CartPole

Attempting to AI

Using OpenAI and Farama Gymnasium [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## v0/Random moves

### Average score: 22.2726

This version is just random movements being made so that I could learn about gymnasium and how to use it. Very first try with AI!

## v1/Human code

### Average score of v1.1: 42.7233

v1.1 was the very first attempt at trying to control the environment. It chooses it's movements based on which side the pendulum is currently on trying to counterbalance.

### Average score of v.1.2: 500

v1.2 came after I realized that the score of v1.1 was bad was because it swings too fast. Therefore this iteration tries to not swing too fast while staying upright.

## v2/REINFORCE AI

### Average score after 5000 episodes: 395.72

This version I followed the REINFORCE tutorial on Farama's Gymnasium documentation. Their solution was using Mujoco's Inverted Pendulum while I was using CartPole so I had to make minor tweaks to make it run correctly.
