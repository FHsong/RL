import numpy as np
import time, copy, random
import pandas as pd
from HeadFirst_chapter5.maze_env import Maze


DISCOUNT_FACTOR = 0.9

class SarsaTable:
    def __init__(self, env, learning_rate, discount_factor, epsilon):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {} # 键值为状态，其值为动作概率


    def choose_action_e_greedy(self, state):
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon: # 贪婪条件满足，则选择最佳动作策略
            action = self.getRandomAction_withSame_Q(state)
        else:  # 随机选择动作
            action = np.random.choice(self.env.action_space)
        return action


    # 如果该状态的所有q值都是相等的，那么当选择最大的值的index时，总会选择第一个
    # 编写函数，若有相同的最大值，则随机返回最大值中对应的动作
    def getRandomAction_withSame_Q(self, current_state):
        action_Q_list = self.q_table[current_state]

        max_Q_index = []
        max_Q = np.max(action_Q_list)
        for i in range(len(action_Q_list)):
            if action_Q_list[i] == max_Q:
                max_Q_index.append(i)
        return random.choice(max_Q_index)


    def learn(self, current_state, action, r, next_state):
        self.check_state_exist(next_state)
        q_predict = self.q_table[current_state][action]
        if next_state not in env.terminate_space:
            next_state_action = self.choose_action_e_greedy(next_state)
            q_target = r + self.gamma * (self.q_table[next_state][next_state_action])
        else:
            q_target = r
        self.q_table[current_state][action] += self.lr * (q_target - q_predict)


    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.] * len(env.action_space)



    def show_q_table(self, q_table):
        print('\n----------------- q_table -------------------')
        i = 1
        for key, action in q_table.items():
            print(key, ':  ', action)


if __name__ == '__main__':
    learning_rate = 0.01
    discount_factor = 0.9
    epsilon = 0.7

    env = Maze()
    Sarsa = SarsaTable(env, learning_rate, discount_factor, epsilon)

    for episode in range(200):
        env.current_state = env.reset()
        env.render()
        while True:
            action = Sarsa.choose_action_e_greedy(env.current_state)

            next_state, r, done = env.step(action)

            Sarsa.learn(env.current_state, action, r, next_state)

            env.current_state = next_state

            env.render()

            if done:
                time.sleep(0.5)
                break

            # RL.show_q_table(RL.q_table)




