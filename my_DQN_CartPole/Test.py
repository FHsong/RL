import gym
from my_DQN_CartPole.DQN import DQN
from my_DQN_CartPole.DQN_ER import DQN_ER
from my_DQN_CartPole.DoubleDQN_ER import DoubleDQN_ER

env = gym.make('CartPole-v1')  # 创建游戏环境
learning_rate = 0.001
n_episode = 200
batch = 128

result_path = 'ExperimentResults/'

dqn = DQN(env=env, learning_rate=learning_rate, n_episode=n_episode)
dqn_ER = DQN_ER(env=env, learning_rate=learning_rate, n_episode=n_episode, batch=batch)
double_dqn_ER = DoubleDQN_ER(env=env, learning_rate=learning_rate, n_episode=n_episode, batch=batch)

# dqn.train(result_path + 'DQN')
dqn_ER.train(result_path + 'DQN_ER')
double_dqn_ER.train(result_path + 'DoubleDQN')


