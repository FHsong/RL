import gym
import time
from myHeadFirst_chapter2.grid_mdp import GridEnv

# env = gym.make('GridWorld-v0')

env = GridEnv()
env.reset()
env.render()

time.sleep(3)
env.close()

