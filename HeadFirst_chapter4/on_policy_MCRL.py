import numpy as np
import time, copy, random
import pandas as pd
from HeadFirst_chapter4.maze_Env import MazeEnv


DISCOUNT_FACTOR = 0.9

class Agent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.5
        self.epsilon = 0.001 # ε-soft策略
        # 前4列是π(a|s) (即是选择每个动作的概率)，且初始化为0.25，意味着当前状态选择4个动作的概率都是0.25。
        # 最后一列为该状态的v-value
        self.pi_V = {}
        for state in self.env.state_space:
            self.pi_V[state] = [0.25, 0.25, 0.25, 0.25, 0.]


        self.sample_num = 1000
        self.state_sample = None
        self.action_sample = None
        self.reward_sample = None



    def ES_MC_Algorithm(self, gamma, pi_V, sample_num):
        for I in range(sample_num):
            #------------------- 生成一个episode 样本-------------------
            state_episode = []
            action_episode = []
            reward_episode = []
            # 不产生终止状态
            current_state = random.choice(list(self.env.state_space))
            while current_state in self.env.terminate_space:
                current_state = random.choice(list(self.env.state_space))

            done = False
            count = 0
            while done == False and count < 100 :
                self.env.state = current_state
                action = self.RWS(pi_V[current_state][0:4])  # 初始采样采用随机的策略来选择动作
                next_state, r, done = self.env.step(action)
                state_episode.append(current_state)
                action_episode.append(action)
                reward_episode.append(r)
                current_state = next_state
                count += 1
            #------------------- 生成一个episode 样本 -------------------

            # print('The ', I+1, '-th episode----')
            # self.print_episode(state_episode, action_episode, reward_episode)

            # ------------------- 计算episode中出现转态的价值函数 -------------------
            V_value = {} # 关键字为状态，其值为该状态的价值函数
            N_value = {} # 对应的转态出现了几次
            for state in state_episode:
                V_value[state] = 0.
                N_value[state] = 0.

            G = 0.
            for step in range(len(state_episode) - 1, -1, -1):
                G *= gamma
                G += reward_episode[step]
            for step in range(len(state_episode)):
                s = state_episode[step]
                V_value[s] += G
                N_value[s] += 1.
                G -= reward_episode[step]
                G /= gamma
            for s in V_value:
                V_value[s] /= N_value[s]
            # ------------------- 计算episode中出现转态的价值函数 -------------------

            # ------------------- 根据以上的价值函数改进当前的策略π(a|s) -------------------
            # 首先根据以上的V_value来更新全局的pi_V
            for state in self.env.state_space:
                if state in state_episode:
                    pi_V[state][4] = V_value[state]

            # 贪婪选择
            for state in state_episode: # 针对当前episode中的每个状态
                self.env.state = state
                action_1 = self.env.action_space[0]
                next_state, r, done = self.env.step(action_1)
                value_1 = r + gamma * pi_V[next_state][4]

                for action in self.env.action_space:
                    next_state, r, done = self.env.step(action)
                    if value_1 < r + gamma * pi_V[next_state][4]:
                        action_1 = action
                        value_1 = r + gamma * pi_V[next_state][4]

                for temp_action in self.env.action_space:
                    if temp_action != action_1:
                        pi_V[state][temp_action] = self.epsilon / len(self.env.action_space)
                    else:
                        pi_V[state][temp_action] = 1 - self.epsilon + self.epsilon / len(self.env.action_space)
        return pi_V




    # 根据当前转态s的每个动作的概率π(a|s)来选择一个动作
    # 若是每个概率相等，则是均匀选择。若每个动作概率为1，则总是选它
    def RWS(self, P):
        m = 0
        r = random.random()
        for i in range(len(P)):
            m += P[i]
            if r <= m:
                return i


    def print_episode(self, state_episode, action_episode, reward_episode):
        print('----- The length of the episode', len(state_episode), '-------')

        for state in state_episode:
            print(state, end=' ')
        print('')
        for action in action_episode:
            print(action, end=' ')
        print('')
        for reward in reward_episode:
            print(reward, end=' ')
        print('')



    def show_pi_V(self, pi_V):
        print('\n----------------- pi(a|s) -------------------')
        i = 1
        for y in range(550, 100, -100):
            j = 1
            for x in range(150, 600, 100):
                print(pi_V[(i, j)][0:4], end='   ')
                j += 1
            i += 1
            print('')

    # def show_policy(self, pi):
    #
    # def


if __name__ == '__main__':
    env = MazeEnv()
    env.reset()
    env.render()
    time.sleep(0.5)

    sample_num = 1000
    agent = Agent(env)
    pi_V = agent.ES_MC_Algorithm(agent.gamma, agent.pi_V, sample_num)

    agent.show_pi_V(pi_V)

    env.state = (4, 4)
    env.render()
    while True:
        action = np.argmax(pi_V[env.state][0:4])  # env.state 是当前状态
        next_state = env.transition[(env.state[0], env.state[1], action)]
        env.state = next_state
        env.render()
        time.sleep(0.5)
        if env.state is env.treasure_space:
            break

    print('AAAA')
    print('\n')

    # 防止生成终止状态
