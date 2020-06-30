import  gym,os, matplotlib
import  numpy as np
import  tensorflow as tf
from    tensorflow.keras import layers,optimizers,losses, Model
from    collections import namedtuple
from matplotlib import pyplot as plt

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['figure.figsize'] = [8, 5]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

#####################  hyper parameters  ####################
lambd = .9             # 折扣率(decay factor)
render = False          # 是否渲染游戏


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = layers.Dense(32, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(16, kernel_initializer='he_normal', activation='relu')
        self.out = layers.Dense(2, kernel_initializer='he_normal', activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.out(x)
        return x


class DQN():
    def __init__(self, env, learning_rate, n_episode):
        self.DQN = Network()

        self.env = env
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate)  # 优化器
        self.n_episode = n_episode
        self.epsilon = 0.1


    def get_Q_value(self, state):
        state = tf.constant(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        Q_value = self.DQN(state).numpy()
        return Q_value


    def run(self, result_path):
        returns = []
        total_reward = 0
        for episode in range(self.n_episode):
            state = self.env.reset()
            for step in range(500):
                # if render == True: env.render()

                allQ = self.get_Q_value(state)
                action = np.argmax(allQ, axis=1)[0]
                # e-Greedy：如果小于epsilon，就智能体随机探索。否则，就用最大Q值的动作。
                if np.random.random() > self.epsilon:
                    action = self.env.action_space.sample()

                next_state, r, done, _ = self.env.step(action)

                Q1 = self.get_Q_value(next_state)

                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, action] = r + lambd * maxQ1

                with tf.GradientTape() as tape:
                    state = tf.constant(state, dtype=tf.float32)
                    state = tf.expand_dims(state, axis=0)
                    _qvalues = self.DQN(state)

                    targetQ = tf.constant(targetQ, dtype=tf.float32)
                    _loss = tf.losses.mean_squared_error(targetQ, _qvalues)

                grad = tape.gradient(_loss, self.DQN.trainable_weights)
                self.optimizer.apply_gradients(zip(grad, self.DQN.trainable_weights))

                state = next_state
                total_reward += r
                if done:
                    # self.epsilon = 1. / ((episode / 50) + 10)
                    break

            if episode % 20 == 0:
                returns.append(total_reward / 20)
                total_reward = 0
                print('Episode: {}/{}  | Episode Average Reward: {:.4f}'
                      .format(episode, self.n_episode, returns[-1]))

        self.env.close()
        print(np.array(returns))
        plt.figure()
        plt.plot(np.arange(len(returns)) * 20, np.array(returns))
        plt.plot(np.arange(len(returns)) * 20, np.array(returns), 's')
        plt.title('DQN')
        plt.xlabel('回合数')
        plt.ylabel('总回报')
        plt.savefig(result_path)
        # plt.show()







