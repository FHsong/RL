import numpy as np
import time, copy, random
import pandas as pd
from HeadFirst_chapter3.maze_Env import MazeEnv


DISCOUNT_FACTOR = 0.9

class Agent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.9
        self.pi_V = {} # 前4列是π(a|s)，且初始化为0.25，意味着当前状态选择4个动作的概率都是0.25。最后一列为该状态的v-value
        for state in self.env.state_space:
            self.pi_V[state] = [0.25, 0.25, 0.25, 0.25, 0.]



    def value_iteration(self, gamma, pi_V):
        theta = 0.0001
        for _ in range(1000):
            delta = 0.
            for state in self.env.state_space:
                if state not in self.env.terminate_space:
                    self.env.state = state
                    action_1 = self.env.action_space[0]
                    next_state, r, done = self.env.step(action_1)
                    value_1 = r + gamma * pi_V[next_state][4]

                    for action in self.env.action_space:
                        next_state, r, done = self.env.step(action)
                        if value_1 < r + gamma * pi_V[next_state][4]:
                            action_1 = action
                            value_1 = r + gamma * pi_V[next_state][4]

                    delta =  max(delta, np.abs(pi_V[state][4] - value_1))

                    # 下面for循环相当于执行 self.pi_V[state] = action_1
                    for temp_action in self.env.action_space:
                        if temp_action != action_1:
                            pi_V[state][temp_action] = 0
                        else:
                            pi_V[state][temp_action] = 1
                    pi_V[state][4] = value_1

            if delta < theta:
                break

        return pi_V



    def show_pi_V(self, pi_V):
        print('\n----------------- V-value -------------------')
        i = 1
        for y in range(550, 100, -100):
            j = 1
            for x in range(150, 600, 100):
                print('%.4f' %pi_V[(i, j)][4], end='   ')
                j += 1
            i += 1
            print('')


if __name__ == '__main__':
    env = MazeEnv()
    env.reset()
    env.render()
    time.sleep(0.5)


    agent = Agent(env)

    is_stable = False
    pi_V = None
    i = 0
    while i<100 :
        before_pi_V = copy.deepcopy(agent.pi_V)
        pi_V = agent.value_iteration(agent.gamma, agent.pi_V)

        for key, element in pi_V.items():
            if key not in env.terminate_space:
                print(key, ': ', element[0:4])
        print('\n')
        for state in before_pi_V:
            if state not in env.terminate_space:
                if pi_V[state][0:4] != before_pi_V[state][0:4]:
                    is_stable = False
        i+=1

    env.state = (1, 3)
    env.render()
    while True:
        action = np.argmax(pi_V[env.state][0:4])  # env.state 是当前状态
        next_state = env.transition[(env.state[0], env.state[1], action)]
        env.state = next_state
        env.render()
        time.sleep(0.5)
        if env.state is env.treasure_space:
            break

    env.close()