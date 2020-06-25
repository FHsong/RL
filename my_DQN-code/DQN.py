import  gym,os, matplotlib
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
from    collections import namedtuple
from matplotlib import pyplot as plt

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['figure.figsize'] = [8, 5]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

#####################  hyper parameters  ####################
lambd = .9             # 折扣率(decay factor)
epsilon = 0.1                 # epsilon-greedy算法参数，越大随机性越大，越倾向于探索行为。
render = False          # 是否渲染游戏

env = gym.make('CartPole-v1')  # 创建游戏环境

class DQN(keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(32, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(16, kernel_initializer='he_normal', activation='relu')
        self.out = layers.Dense(2, kernel_initializer='he_normal', activation=None)


    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.out(x)
        return x


    def get_Q_value(self, state):
        state = tf.constant(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        Q_value = self.call(state).numpy()
        return Q_value


num_episodes = 200    # 迭代次数
if __name__ == '__main__':
    agent = DQN()
    optimizer = optimizers.Adam(learning_rate=0.01)
    returns = []
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(500):
            if render == True: # 渲染
                env.render()

            allQ = agent.get_Q_value(state)
            action = np.argmax(allQ, axis=1)[0]
            # e-Greedy：如果小于epsilon，就智能体随机探索。否则，就用最大Q值的动作。
            if np.random.random() < epsilon:
                action = env.action_space.sample()

            next_state, r, done, _ = env.step(action)

            Q1 = agent.get_Q_value(next_state)

            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, action] = r + lambd * maxQ1

            with tf.GradientTape() as tape:
                state = tf.constant(state, dtype=tf.float32)
                state = tf.expand_dims(state, axis=0)
                _qvalues = agent(state)

                targetQ = tf.constant(targetQ, dtype=tf.float32)
                _loss = tf.losses.MSE(targetQ, _qvalues)

            grad = tape.gradient(_loss, agent.trainable_weights)
            optimizer.apply_gradients(zip(grad, agent.trainable_weights))

            state = next_state
            total_reward += r
            if done:
                break

        if episode % 20 == 0:
            returns.append(total_reward / 20)
            total_reward = 0
            print('Episode: {}/{}  | Episode Average Reward: {:.4f}'
                  .format(episode, num_episodes, returns[-1]))

    env.close()
    print(np.array(returns))
    plt.figure()
    plt.plot(np.arange(len(returns)) * 20, np.array(returns))
    plt.plot(np.arange(len(returns)) * 20, np.array(returns), 's')
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.show()







