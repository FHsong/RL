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


    def policy_evaluation(self, gamma, pi_V):
        theta = 0.0001
        for _ in range(1000):
            delta = 0
            for state in self.env.state_space:
                if state not in self.env.terminate_space:
                    self.env.state = state
                    new_v = 0.
                    for action in self.env.action_space:
                        next_state, r, done = self.env.step(action)
                        new_v += pi_V[state][action] * (r + gamma * pi_V[next_state][4])

                    delta = max(delta, np.abs(pi_V[state][4]) - new_v)
                    self.pi_V[state][4] = new_v
            if delta < theta:
                break


    def policy_improvement(self, gamma, pi_V):
        for state in self.env.state_space:
            if state not in self.env.terminate_space:
                self.env.state = state
                action_1 = self.env.action_space[0]
                next_state, r, done = self.env.step(action_1)

                v_value = r + gamma * pi_V[next_state][4]
                for action in self.env.action_space:
                    next_state, r, done = self.env.step(action)
                    if v_value < (r + gamma * pi_V[next_state][4]):
                        action_1 = action
                        v_value = (r + gamma * pi_V[next_state][4])

                for temp_action in self.env.action_space:
                    if temp_action != action_1:
                        pi_V[state][temp_action] = 0
                    else:
                        pi_V[state][temp_action] = 1


    def optimize(self):
        is_stable = False
        round_num = 0

        while not is_stable:
            is_stable = True
            print("\nRound Number:" + str(round_num))
            round_num += 1

            self.policy_evaluation(self.gamma, self.pi_V)
            print("Expected Value according to Policy Evaluation")

            before_pi_V = copy.deepcopy(self.pi_V)
            self.policy_improvement(self.gamma, self.pi_V)
            self.show_pi_V(self.pi_V)

            for state in before_pi_V:
                if state not in self.env.terminate_space:
                    if self.pi_V[state][0:4] != before_pi_V[state][0:4]:
                        is_stable = False


        return self.pi_V



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
    pi_V = agent.optimize()

    env.state = (1,1)
    env.render()
    while True:
        action = np.argmax(pi_V[env.state][0:4])  # env.state 是当前状态
        next_state = env.transition[(env.state[0], env.state[1], action)]
        env.state = next_state
        env.render()
        time.sleep(0.5)
        if env.state is env.treasure_space:
            break


    # for key, element in pi_V.items():
    #     if key not in env.terminate_space:
    #         print(key, ': ', element[0:4])

