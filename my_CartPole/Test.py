import gym
from my_CartPole.a___DQN import DQN
from my_CartPole.b__DQN_ER import DQN_ER
from my_CartPole.c__DoubleDQN_ER import DoubleDQN_ER
from my_CartPole.DuelingDQN_ER import DuelingDQN_ER
from my_CartPole.d__PG import PolicyGradient



env = gym.make('CartPole-v1')  # 创建游戏环境
learning_rate = 0.01
n_episode = 200
batch = 128

result_path = 'ExperimentResults/'

# dqn = DQN(env=env, learning_rate=learning_rate, n_episode=n_episode)
# dqn_ER = DQN_ER(env=env, learning_rate=learning_rate, n_episode=n_episode, batch=batch)
# double_dqn_ER = DoubleDQN_ER(env=env, learning_rate=learning_rate, n_episode=n_episode, batch=batch)
#

# dqn.run(result_path + 'DQN')
# dqn_ER.run(result_path + 'DQN_ER')
# double_dqn_ER.run(result_path + 'DoubleDQN')

policy_gradient = PolicyGradient(env=env, learning_rate=learning_rate, n_episode=n_episode)
policy_gradient.run()



